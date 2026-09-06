#!/usr/bin/env python3
"""embed_extract_hf.py — extract the embedding table / lm_head weight from a HF
causal-LM checkpoint via the transformers API.

Same output contract as yecode/embed/freertos-yeats/applications/fpga_test/
translation_test/tools/embed_extract.py (bare row-major, no header, so the NPU
can point a basereg at row 0) but loads through AutoModelForCausalLM instead of
parsing the safetensors header by hand. Needs transformers + torch on the host;
use that other tool where only a C compiler is available.

-m/--model-dir points at the checkpoint to read. Examples below use QWEN, e.g.
QWEN=$HF/models/llm/Qwen2.5-0.5B-Instruct.

    # full fp16 input-embedding table: 151936 x 896, 272 MB
    embed_extract_hf.py -m $QWEN -o embed.f16

    # lm_head, transposed to [hidden, vocab] for a row-major GEMM
    embed_extract_hf.py -m $QWEN --which lm_head --transpose -o head.f16

    # compact table: only the rows listed in ids.txt, in that order
    embed_extract_hf.py -m $QWEN -o embed8k.f16 --rows ids.txt

    # keep the checkpoint's bf16 bits, pad to SD block size, self-check
    embed_extract_hf.py -m $QWEN -o embed.bf16 --dtype bf16 --align 512 --verify

Qwen2.5 sets tie_word_embeddings, so model.embed_tokens.weight *is* the lm_head:
--which embed and --which lm_head hand back the same storage, and the tool says
so instead of letting you believe you pulled two independent matrices.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

CHUNK_ROWS = 8192                      # 8192 x 896 fp32 = 28 MB per block
NP_DTYPE = {"bf16": np.uint16, "fp16": np.float16, "fp32": np.float32}

def load_tables(model_dir: str):
    """Load the checkpoint on CPU in its stored dtype and hand back both tables.

    device_map is left alone on purpose: this is a weight dump, nothing runs, and
    an accelerate offload would only add meta tensors we would have to fetch back.
    """
    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="auto")
    model.eval()

    embed = model.get_input_embeddings().weight
    head_mod = model.get_output_embeddings()
    head = head_mod.weight if head_mod is not None else None

    # Trust storage identity, not the config flag: a checkpoint can carry a real
    # separate lm_head even when tie_word_embeddings is set in config.json.
    tied = head is None or head.data_ptr() == embed.data_ptr()
    return cfg, embed.detach(), (embed.detach() if head is None else head.detach()), tied


def parse_rows(path: str, vocab: int) -> list[int]:
    ids = [int(x) for x in Path(path).read_text().split() if x.strip()]
    if not ids:
        raise SystemExit(f"{path} has no ids")
    bad = [i for i in ids if not 0 <= i < vocab]
    if bad:
        raise SystemExit(f"{len(bad)} id(s) outside [0,{vocab}), e.g. {bad[:5]}")
    return ids


def as_numpy(t: torch.Tensor, dtype: str) -> np.ndarray:
    """torch 2D chunk -> numpy in the requested on-disk dtype.

    bf16 is bit-preserving: numpy has no bfloat16, so the tensor is reinterpreted
    as uint16 and the raw little-endian halves go straight to disk. fp16/fp32 go
    through torch's rounding, which is correct-rounding for bf16->fp16 (both have
    an 8-bit exponent, so the only loss is on subnormals/overflow, of which bf16
    values inside fp16 range have none).
    """
    t = t.contiguous()
    if dtype == "bf16":
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        return t.view(torch.uint16).numpy()
    return t.to(torch.float16 if dtype == "fp16" else torch.float32).numpy()


def write_table(src: torch.Tensor, out: Path, dtype: str, align: int):
    """Stream src to disk row-block by row-block, returning (rows, cols, stats).

    Chunked so a 272 MB table never needs a second full-size copy in RAM, and so
    the fp32 stats pass rides along on data already in cache.
    """
    rows, cols = src.shape
    lo, hi, abs_sum, sha = np.inf, -np.inf, 0.0, hashlib.sha256()

    with out.open("wb") as f:
        for start in range(0, rows, CHUNK_ROWS):
            blk = as_numpy(src[start:start + CHUNK_ROWS], dtype)
            f.write(blk.tobytes())
            sha.update(blk.tobytes())
            fp = src[start:start + CHUNK_ROWS].to(torch.float32)
            lo, hi = min(lo, fp.min().item()), max(hi, fp.max().item())
            abs_sum += fp.abs().sum().item()

        pad = 0
        if align and (f.tell() % align):
            pad = align - (f.tell() % align)
            f.write(b"\x00" * pad)

    return rows, cols, {"min": lo, "max": hi,
                        "absmean": abs_sum / (rows * cols),
                        "sha256": sha.hexdigest(), "pad": pad}


def verify(src: torch.Tensor, out: Path, dtype: str, rows: int, cols: int) -> bool:
    """Re-read the file and byte-compare a spread of rows against the source."""
    itemsize = np.dtype(NP_DTYPE[dtype]).itemsize
    want = rows * cols * itemsize
    got = out.stat().st_size
    if got < want:
        print(f"  verify: FAIL short file {got:,} B < {want:,} B", file=sys.stderr)
        return False

    disk = np.memmap(out, dtype=NP_DTYPE[dtype], mode="r", shape=(rows, cols))
    picks = sorted({0, rows // 2, rows - 1, *np.random.default_rng(0)
                    .integers(0, rows, size=min(8, rows)).tolist()})
    for r in picks:
        ref = as_numpy(src[r:r + 1], dtype)[0]
        if not np.array_equal(disk[r].view(NP_DTYPE[dtype]), ref):
            print(f"  verify: FAIL row {r} differs", file=sys.stderr)
            return False
    print(f"  verify: {len(picks)} sampled rows byte-identical, size {got:,} B ok")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model-dir", required=True,
                    help="HF model dir (config.json + weights)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--which", choices=("embed", "lm_head"), default="embed",
                    help="input embedding table or output projection (default: embed)")
    ap.add_argument("--dtype", choices=tuple(NP_DTYPE), default="fp16",
                    help="on-disk dtype (default: fp16; bf16 keeps checkpoint bits)")
    ap.add_argument("--transpose", action="store_true",
                    help="write [hidden, vocab] instead of [vocab, hidden]")
    ap.add_argument("--rows", metavar="FILE",
                    help="extract only these token ids, in this order (one per line); "
                         "writes <output>.idmap.txt so the target can map compact "
                         "index -> real token id")
    ap.add_argument("--limit", type=int, default=0, help="extract only the first N rows")
    ap.add_argument("--align", type=int, default=0, metavar="N",
                    help="zero-pad the file up to a multiple of N bytes (e.g. 512)")
    ap.add_argument("--npy", action="store_true",
                    help="write a .npy instead of bare bytes (fp16/fp32 only)")
    ap.add_argument("--verify", action="store_true", help="re-read and byte-compare rows")
    args = ap.parse_args()

    if args.rows and args.limit:
        raise SystemExit("--rows and --limit are mutually exclusive")
    if args.npy and args.dtype == "bf16":
        raise SystemExit("--npy needs fp16/fp32: numpy has no bfloat16")

    cfg, embed, head, tied = load_tables(args.model_dir)
    table = embed if args.which == "embed" else head
    vocab, hidden = table.shape
    print(f"{args.which}: {table.dtype} [{vocab}, {hidden}] "
          f"(config vocab_size={getattr(cfg, 'vocab_size', '?')}, "
          f"hidden_size={getattr(cfg, 'hidden_size', '?')})")
    if tied:
        print("  tie_word_embeddings: embed and lm_head are the same tensor "
              "(lm_head applies it transposed)")

    ids = None
    if args.rows:
        ids = parse_rows(args.rows, vocab)
    elif args.limit:
        if args.limit > vocab:
            raise SystemExit(f"--limit {args.limit} > {vocab} rows")
        ids = list(range(args.limit))
    if ids is not None:
        table = table.index_select(0, torch.tensor(ids, dtype=torch.long))
        print(f"  row subset: {len(ids)} of {vocab} rows")

    src = table.t() if args.transpose else table
    out = Path(args.output)

    if args.npy:
        np.save(out, as_numpy(src, args.dtype))
        out = out.with_suffix(".npy") if out.suffix != ".npy" else out
        print(f"wrote {out} ({out.stat().st_size:,} B, .npy header included)")
        return 0

    rows, cols, st = write_table(src, out, args.dtype, args.align)
    size = out.stat().st_size
    print(f"wrote {out} [{rows}, {cols}] {args.dtype} "
          f"({size:,} B = {(size + 511) // 512} SD blocks"
          + (f", {st['pad']} B pad to {args.align}" if st["pad"] else "") + ")")
    print(f"  range [{st['min']:.6g}, {st['max']:.6g}]  mean|w| {st['absmean']:.6g}")
    print(f"  sha256 {st['sha256']}")
    print(f"  set TRANS_EMBED_ROWS={rows} TRANS_EMBED_SD_BLOCKS={(size + 511) // 512}")

    if args.rows:
        idmap = Path(str(out) + ".idmap.txt")
        idmap.write_text("\n".join(str(i) for i in ids) + "\n")
        print(f"  id map -> {idmap} (compact index i holds token id ids[i])")

    if args.verify and not verify(src, out, args.dtype, rows, cols):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
