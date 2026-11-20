#!/usr/bin/env python3
"""Inference a TensorRT engine by feeding one input binary file.

Modeled after ai_infer_onnx.py (same CLI style: -model/-name/-bin/-fmt/-v),
but for a serialized TensorRT engine (.engine/.plan) instead of ONNX.

    python ai_infer_trt_engine.py -model lenet.plan -bin input.bin -fmt float32
    python ai_infer_trt_engine.py -model lenet.plan -bin input.bin -fmt uint8 -v
    python ai_infer_trt_engine.py -model lenet.plan -iters 5          # random input
    python ai_infer_trt_engine.py -model lenet.plan -iters 5 -time    # + timing

Use -plugin /path/to/plugin.so to load custom TensorRT plugins; repeat the
option to load multiple libraries before deserializing the engine.

Input data is read from -bin (random data if -bin is not given), normalized
to float32, and fed to every engine input (single-input engines like LeNet just
take the whole buffer). Outputs are dumped to PC_<output_name>.bin, mirroring
the ONNX tool's convention. Inference timing (CUDA events, 1st iter warmup
excluded) is only measured with -time.
"""
import os
import sys
import argparse
import ctypes

import numpy as np
import tensorrt as trt
import cuda.bindings.runtime as cudart

dtype_str = "<float64|float32|float16|uint64|int64|uint32|int32|uint16|int16|uint8|int8>"

NP_DTYPE = {
    "float64": np.float64, "float32": np.float32, "float16": np.float16,
    "uint64": np.uint64, "int64": np.int64, "uint32": np.uint32,
    "int32": np.int32, "uint16": np.uint16, "int16": np.int16,
    "uint8": np.uint8, "int8": np.int8,
}

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def cu(res):
    err = res[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return res[1] if len(res) > 1 else None


def trt_dtype_to_np(t):
    return np.frombuffer(trt.nptype(t).tobytes(), dtype=trt.nptype(t)).dtype


def load_input(args, shape, np_dtype, rng=None):
    """Read the raw binary (or generate random data when -bin is not given),
    cast to the engine input dtype, reshape."""
    n_elems = int(np.prod(shape))
    if not args.bin:
        if rng is None:
            rng = np.random.default_rng(0)
        if np.issubdtype(np_dtype, np.floating):
            data = rng.standard_normal(tuple(shape)).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.integer) or np.issubdtype(np_dtype, np.bool_):
            data = rng.integers(1, 10, size=tuple(shape)).astype(np_dtype)
        else:
            raise SystemExit(f"unsupported random input dtype: {np_dtype}")
        print(" ==> Feed TRT engine input: (random)", shape)
        return data

    print(" ==> Feed TRT engine input ", shape, ", file: ", args.bin)
    raw = np.fromfile(args.bin, dtype=NP_DTYPE[args.fmt])
    if raw.size < n_elems:
        raise SystemExit(
            f"input file {args.bin} has {raw.size} elems, engine needs {n_elems}")
    data = raw[0:n_elems].astype(np_dtype)
    # normalize like ai_infer_onnx.py does for uint8 input
    if args.fmt == "uint8":
        data = data / 256.
    return data.reshape(shape)


def inference_engine(args):
    plugin_handles = []
    for plugin in args.plugin:
        try:
            plugin_handles.append(ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL))
        except OSError as error:
            raise SystemExit(f"failed to load plugin {plugin}: {error}") from error
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(args.model, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise SystemExit(f"failed to deserialize engine: {args.model}")
    context = engine.create_execution_context()
    if context is None:
        raise SystemExit(
            "failed to create execution context; check the TensorRT errors above "
            "and use -plugin /path/to/plugin.so for required custom plugins")

    if args.v:
        print("=== Engine I/O ===")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            print(f"  [{i}] {name}  {mode}  shape={tuple(shape)}  dtype={dtype}")

    # collect inputs / outputs in engine order
    in_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i))
                == trt.TensorIOMode.INPUT]
    out_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                 if engine.get_tensor_mode(engine.get_tensor_name(i))
                 == trt.TensorIOMode.OUTPUT]

    # dynamic shapes: resolve -1 dims from the optimization profile (opt shape).
    # Engines may have MULTIPLE dynamic inputs sharing one profile dim (e.g.
    # pi05 prefix.engine: inputs_embeds/rope_rotary_cos_sin/attention_pos_id all
    # bind the same seq_len), so set every dynamic input, not just the first.
    for name in in_names:
        shape = engine.get_tensor_shape(name)
        if any(d == -1 for d in shape):
            profile = engine.get_tensor_profile_shape(name, 0)
            opt_shape = profile[1]  # (min, opt, max) -> use opt
            print(" ==> Dynamic input {} {} -> opt profile {}".format(
                name, tuple(shape), tuple(opt_shape)))
            context.set_input_shape(name, opt_shape)

    # input data: one binary feeds the (first) input
    in_name = in_names[0]
    in_shape = context.get_tensor_shape(in_name)
    in_dtype = trt.nptype(engine.get_tensor_dtype(in_name))
    rng = np.random.default_rng(0)
    input_data = load_input(args, in_shape, in_dtype, rng)

    # device buffers for every tensor (re-read shapes: outputs may depend on
    # the resolved input shape)
    d_addrs = {}
    host_outs = {}
    stream = cu(cudart.cudaStreamCreate())
    for name in in_names + out_names:
        shape = context.get_tensor_shape(name)
        if any(d == -1 for d in shape):
            raise SystemExit(
                f"tensor {name} still has unresolved dynamic shape {tuple(shape)}")
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        d = cu(cudart.cudaMalloc(max(nbytes, 1)))
        d_addrs[name] = d
        if not context.set_tensor_address(name, d):
            raise SystemExit(f"failed to bind tensor {name}")
        if name in in_names and name != in_name and nbytes:
            if np.issubdtype(dtype, np.floating):
                data = rng.standard_normal(tuple(shape)).astype(dtype)
            elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
                data = rng.integers(1, 10, size=tuple(shape)).astype(dtype)
            else:
                raise SystemExit(f"unsupported random input dtype for {name}: {dtype}")
            cu(cudart.cudaMemcpy(d, data.ctypes.data, data.nbytes,
                                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        if name in out_names:
            host_outs[name] = np.empty(shape, dtype=dtype)

    # H2D
    if input_data.nbytes:
        cu(cudart.cudaMemcpy(d_addrs[in_name], input_data.ctypes.data,
                             input_data.nbytes,
                             cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

    # inference. With -time, measure with CUDA events (GPU time, excludes
    # launch overhead). First iteration is warmup (cold L2 / weight
    # first-touch) and is NOT counted in the reported time.
    # warmup: 1 uncounted iteration
    if not context.execute_async_v3(stream):
        raise SystemExit("TensorRT warmup failed; check the TensorRT errors above")
    cu(cudart.cudaStreamSynchronize(stream))

    if args.time:
        err, start = cudart.cudaEventCreate()
        cu((err,))
        err, stop = cudart.cudaEventCreate()
        cu((err,))
        cu((cudart.cudaEventRecord(start, stream)[0],))
        for _ in range(args.iters):
            if not context.execute_async_v3(stream):
                raise SystemExit("TensorRT inference failed; check the TensorRT errors above")
        cu((cudart.cudaEventRecord(stop, stream)[0],))
        cu(cudart.cudaStreamSynchronize(stream))
        err, total_ms = cudart.cudaEventElapsedTime(start, stop)
        cu((err,))
        print(" ==> Inference time: {:.3f} ms total / {} iters = {:.3f} ms/iter "
              "(1st iter warmup, not counted)".format(
                  total_ms, args.iters, total_ms / args.iters))
        cu((cudart.cudaEventDestroy(start)[0],))
        cu((cudart.cudaEventDestroy(stop)[0],))
    else:
        for _ in range(args.iters):
            if not context.execute_async_v3(stream):
                raise SystemExit("TensorRT inference failed; check the TensorRT errors above")
        cu(cudart.cudaStreamSynchronize(stream))

    print("Inference done: {} iters".format(args.iters))

    # D2H + dump, mirroring "PC_<name>.bin" of the ONNX tool
    # (random-input mode skips the dump: profiling use case, no output needed)
    for name in out_names:
        if not args.bin:
            break
        cu(cudart.cudaMemcpy(host_outs[name].ctypes.data, d_addrs[name],
                             host_outs[name].nbytes,
                             cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        out_path = "PC_{}.bin".format(name)
        host_outs[name].tofile(out_path)
        print(" ==> Gen TRT PC output: ", name)
        if args.v:
            print("     shape:", host_outs[name].shape,
                  " dtype:", host_outs[name].dtype)
            print("     values:", host_outs[name].flatten())

    for d in d_addrs.values():
        cu(cudart.cudaFree(d))
    cu(cudart.cudaStreamDestroy(stream))


def init_param(args):
    parser = argparse.ArgumentParser(
        description="Inference TensorRT engine by feeding one input file. Version: 1.0.0")
    parser.add_argument("-model", type=str, required=True, default="model.plan",
                        help="TensorRT engine file (.engine/.plan)")
    parser.add_argument("-plugin", action="append", default=[],
                        type=os.path.abspath,
                        help="custom TensorRT plugin library (.so); repeat for multiple libraries")
    parser.add_argument("-bin", type=str, required=False, default="",
                        help="input binary file (random input if not given)")
    parser.add_argument("-fmt", type=str, required=False, default="uint8",
                        help="input binary format: " + dtype_str)
    parser.add_argument("-iters", type=int, required=False, default=1,
                        help="inference iterations")
    parser.add_argument("-time", action="store_true", required=False,
                        help="measure inference time with CUDA events "
                             "(1st iter warmup, not counted)")
    parser.add_argument("-v", action="store_true", required=False,
                        help="Show verbose info")
    return parser.parse_args(args)


if __name__ == "__main__":
    args = init_param(sys.argv[1:])
    inference_engine(args)
