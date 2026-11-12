#!/usr/bin/env python3
"""Per-kernel external DRAM read/write statistics for a TensorRT engine.

NVIDIA Thor (Tegra, unified memory) edition. The SM-side dram__* counters are
n/a on Thor; the working path is the MCC (memory controller) throughput
counters read by the STANDALONE Nsight Compute 2026.4:

    mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed   [%]
    mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed  [%]

These are PERCENTAGES of the LPDDR5X peak (273 GB/s on Thor), so bytes are
recovered as:  bytes = kernel_time[s] * pct/100 * 273 GB/s.

This tool wraps the whole flow (modeled after ai_infer_trt_engine.py):

    python ai_dram_statis_trt_engine.py -model lenet.plan -iters 3
    python ai_dram_statis_trt_engine.py -model visual.engine -iters 3 -csv out.csv

Flow:
  1. locate a standalone ncu that can read MCC counters (probe first)
  2. run `ncu --metrics ... <this file> --worker ...` as the target
  3. parse the raw CSV, convert pct -> MB per kernel
  4. print per-kernel table + per-inference summary, optionally save CSV

Notes:
  - sudo is required (ERR_NVGPUCTRPERM); pass SUDO_PW=<pw> to avoid the prompt
  - --cache-control none (default) keeps realistic cache behaviour; "all"
    flushes L2 before every kernel and inflates DRAM reads ~3.5x
  - the first inference under ncu is cold-cache; steady-state numbers need
    -iters >= 3 and reading the per-iteration breakdown
"""
import argparse
import csv as csvmod
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import tensorrt as trt
import cuda.bindings.runtime as cudart

# ---------------------------------------------------------------- constants
PEAK_GBS = 273.0  # Thor LPDDR5X peak, GB/s

DUR = "gpu__time_duration.sum"
MCC_RD = "mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed"
MCC_WR = "mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed"
METRICS = ",".join([DUR, MCC_RD, MCC_WR])

# standalone build first: the CUDA-bundled ncu (2025.3) returns nan for MCC
NCU_CANDIDATES = [
    "/usr/local/NVIDIA-Nsight-Compute/ncu",
    "/usr/local/cuda/bin/ncu",  ## Old Version
]

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def log(msg):
    print("[ncu_dram_statis] {}".format(msg), file=sys.stderr, flush=True)


def cu(res):
    err = res[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError("CUDA error: {}".format(err))
    return res[1] if len(res) > 1 else None


# ------------------------------------------------------------ worker mode
def worker(args):
    """Minimal inference runner executed UNDER ncu (the profiled target).

    Reuses the ai_infer_trt_engine.py logic: deserialize -> resolve dynamic
    shapes -> random input -> warmup 1 -> N timed-free iterations. No printing
    noise, no output dump: ncu only cares about the kernel launches.
    """
    print("--------------------------------")
    print("Runtime Inferece: ", args.model)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(args.model, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise SystemExit("failed to deserialize engine: {}".format(args.model))
    context = engine.create_execution_context()

    in_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i))
                == trt.TensorIOMode.INPUT]
    out_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                 if engine.get_tensor_mode(engine.get_tensor_name(i))
                 == trt.TensorIOMode.OUTPUT]

    # resolve every dynamic input from the opt profile
    for name in in_names:
        shape = engine.get_tensor_shape(name)
        if any(d == -1 for d in shape):
            opt_shape = engine.get_tensor_profile_shape(name, 0)[1]
            context.set_input_shape(name, opt_shape)

    d_addrs = {}
    stream = cu(cudart.cudaStreamCreate())
    for name in in_names + out_names:
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        d = cu(cudart.cudaMalloc(nbytes))
        d_addrs[name] = d
        context.set_tensor_address(name, d)

    # random input, H2D once
    in_name = in_names[0]
    in_shape = context.get_tensor_shape(in_name)
    in_dtype = trt.nptype(engine.get_tensor_dtype(in_name))
    x = np.random.randn(*in_shape).astype(in_dtype)
    cu(cudart.cudaMemcpy(d_addrs[in_name], x.ctypes.data, x.nbytes,
                         cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

    # warmup (uncounted) then the profiled iterations
    context.execute_async_v3(stream)
    cu(cudart.cudaStreamSynchronize(stream))

    for _ in range(args.iters):
        context.execute_async_v3(stream)
    cu(cudart.cudaStreamSynchronize(stream))

    for d in d_addrs.values():
        cu(cudart.cudaFree(d))
    cu(cudart.cudaStreamDestroy(stream))


# ------------------------------------------------------------ ncu plumbing
def sudo_prefix():
    """sudo + PYTHONPATH (pip --user cuda-python is invisible under sudo)."""
    pw = os.environ.get("SUDO_PW") or os.environ.get("SUDO_PASSWORD")
    if pw:
        subprocess.run(["sudo", "-S", "-v"], input=pw + "\n",
                       text=True, capture_output=True)
    import site
    user_site = site.getusersitepackages()
    py_paths = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
    if user_site not in py_paths:
        py_paths.append(user_site)
    return ["sudo", "env", "PYTHONPATH=" + ":".join(py_paths)]


def find_ncu(explicit=None):
    cands = [explicit] if explicit else NCU_CANDIDATES
    for c in cands:
        if not c:
            continue
        path = c if os.path.isabs(c) else (shutil.which(c) or "")
        if path and os.path.exists(path):
            return path
    raise SystemExit("no ncu found; install standalone Nsight Compute 2026.4")


def probe_mcc(ncu_bin, self_py, model):
    """One-kernel probe: can this ncu read MCC counters on this GPU?"""
    cmd = sudo_prefix() + [
        ncu_bin, "--target-processes", "all", "--csv",
        "--cache-control", "none", "--metrics", METRICS,
        "--launch-count", "1",
        self_py, os.path.abspath(__file__), "--worker",
        "-model", model, "-iters", "1",
    ]
    print("CMD:", subprocess.list2cmdline(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout or ""
    if "nan" in out or "n/a" in out:
        return False
    return any(ln.lstrip().startswith('"') for ln in out.splitlines())


def capture(ncu_bin, self_py, args, raw_csv):
    cmd = sudo_prefix() + [
        ncu_bin, "--target-processes", "all", "--csv",
        "--replay-mode", "application",
        "--cache-control", args.cache,
        "--metrics", METRICS,
        self_py, os.path.abspath(__file__), "--worker",
        "-model", args.model, "-iters", str(args.iters),
    ]
    log("run: " + " ".join(cmd[:10]) + " ...")
    with open(raw_csv, "w") as f:
        print("CMD:", subprocess.list2cmdline(cmd))
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        raise SystemExit("ncu failed rc={}: {}".format(
            r.returncode, " | ".join(tail)))


def num(s):
    if s is None:
        return None
    s = s.replace(",", "").strip()
    if s in ("n/a", "(!) n/a", "nan", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_raw(raw_csv):
    """raw ncu CSV (long format) -> list of per-kernel dicts, launch order."""
    with open(raw_csv, newline="") as f:
        lines = [ln for ln in f if ln.lstrip().startswith('"')]
    kernels = {}
    for row in csvmod.DictReader(lines):
        d = kernels.setdefault(row["ID"], {"name": row["Kernel Name"]})
        d[row["Metric Name"]] = num(row["Metric Value"])
    out = []
    for kid in sorted(kernels, key=int):
        d = kernels[kid]
        if d.get(DUR) is None:
            continue
        dur_s = d[DUR] / 1e9
        rd_pct = d.get(MCC_RD) or 0.0
        wr_pct = d.get(MCC_WR) or 0.0
        out.append({
            "id": int(kid),
            "name": d["name"],
            "dur_us": d[DUR] / 1e3,
            "rd_mb": dur_s * (rd_pct / 100 * PEAK_GBS) * 1e3,
            "wr_mb": dur_s * (wr_pct / 100 * PEAK_GBS) * 1e3,
        })
    return out


def short_name(name):
    for key, tag in (("cutlass", "cutlass_conv"), ("sm80_xmma", "xmma_conv"),
                     ("sm50_xmma_pooling", "pooling"), ("__myl_Move", "move"),
                     ("ReshMulSumAdd", "fc/reshape")):
        if key in name:
            return tag
    return name[:24]


def report(kernels, args):
    n_per_iter = None
    # kernels are in launch order: warmup(1) + iters(N); group by 6-ish is
    # engine-specific, so just report warmup vs steady split by counting
    total = len(kernels)
    log("captured {} kernel launches (1 warmup iter + {} profiled iters)".format(
        total, args.iters))

    print("\n== Per-kernel DRAM traffic (first iteration, cold cache) ==")
    print("{:>4} {:<26} {:>9} {:>9} {:>9} {:>9}".format(
        "id", "kernel", "dur_us", "rd_MB", "wr_MB", "tot_MB"))
    for k in kernels:
        print("{:>4} {:<26} {:>9.2f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            k["id"], short_name(k["name"]), k["dur_us"],
            k["rd_mb"], k["wr_mb"], k["rd_mb"] + k["wr_mb"]))

    # aggregate: warmup = first chunk, steady = rest
    rd_all = sum(k["rd_mb"] for k in kernels)
    wr_all = sum(k["wr_mb"] for k in kernels)
    dur_all = sum(k["dur_us"] for k in kernels)
    print("\n== Summary ({} iters, cache-control={}) ==".format(
        args.iters, args.cache))
    print("  kernels          : {}".format(total))
    print("  GPU time         : {:.3f} ms".format(dur_all / 1e3))
    print("  DRAM read        : {:.2f} MB".format(rd_all))
    print("  DRAM write       : {:.2f} MB".format(wr_all))
    print("  DRAM total       : {:.2f} MB".format(rd_all + wr_all))
    if dur_all:
        gbs = (rd_all + wr_all) * 1e3 / dur_all
        print("  avg DRAM bandwidth: {:.1f} GB/s ({:.1f}% of {:.0f} peak)".format(
            gbs, gbs / PEAK_GBS * 100, PEAK_GBS))
    per_iter = (rd_all + wr_all) / max(args.iters, 1)
    print("  per-inference avg: {:.2f} MB (includes 1 cold-cache iteration)".format(
        per_iter))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csvmod.writer(f)
            w.writerow(["kernel_id", "kernel_name", "dur_us",
                        "dram_read_mb", "dram_write_mb", "dram_total_mb"])
            for k in kernels:
                w.writerow([k["id"], k["name"], round(k["dur_us"], 3),
                            round(k["rd_mb"], 4), round(k["wr_mb"], 4),
                            round(k["rd_mb"] + k["wr_mb"], 4)])
        print("\n ==> wrote {}".format(args.csv))


# ---------------------------------------------------------------- main
def init_param(argv):
    ap = argparse.ArgumentParser(
        description="Per-kernel DRAM r/w statistics for a TRT engine (Thor) Version: 1.0.0")
    ap.add_argument("-model", type=str, required=True,
                    help="TensorRT engine file (.engine/.plan)")
    ap.add_argument("-iters", type=int, default=3,
                    help="profiled inference iterations (default 3)")
    ap.add_argument("-csv", type=str, default="",
                    help="save per-kernel table to this CSV")
    ap.add_argument("-ncu", type=str, default="",
                    help="ncu binary (default: auto-detect standalone first)")
    ap.add_argument("-cache", type=str, default="none",
                    choices=["none", "all"],
                    help="--cache-control for ncu (default none=realistic)")
    ap.add_argument("-v", action="store_true", help="verbose")
    ap.add_argument("--worker", action="store_true",
                    help=argparse.SUPPRESS)  # internal: run under ncu
    return ap.parse_args(argv)


def main():
    args = init_param(sys.argv[1:])

    # internal worker mode: executed by ncu, just run the engine
    if args.worker:
        worker(args)
        return

    self_py = sys.executable
    ncu_bin = find_ncu(args.ncu or None)
    log("ncu    = {}".format(ncu_bin))
    log("python = {}".format(self_py))

    log("probing MCC counters with 1 kernel...")
    if not probe_mcc(ncu_bin, self_py, args.model):
        raise SystemExit(
            "{} cannot read mcc__dram_* (all nan/n-a).\n"
            "Use the STANDALONE Nsight Compute 2026.4:\n"
            "  -ncu /usr/local/NVIDIA-Nsight-Compute/ncu".format(ncu_bin))
    log("MCC probe OK (DRAM counters readable)")

    tmpdir = tempfile.mkdtemp(prefix="ai_dram_")
    try:
        raw = os.path.join(tmpdir, "raw.csv")
        capture(ncu_bin, self_py, args, raw)
        kernels = parse_raw(raw)
        if not kernels:
            raise SystemExit("ncu produced no kernel rows (see {})".format(raw))
        report(kernels, args)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
