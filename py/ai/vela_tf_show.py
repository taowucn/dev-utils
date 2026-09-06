#!/usr/bin/env python3
"""Parse the flatbuffer of a Vela-generated tflite directly, to diff against
the FVP hang.

What to look at:
  - Input tensor semantics of the ethos-u custom op
    (command_stream / weights / scratch / scratch_fast).
    Dedicated_Sram mode adds an extra fast scratch tensor; the runtime must
    place it in SRAM and pass the correct base address. If the TFLM side does
    not support that variant, the NPU gets a wrong base address -> never
    raises the completion interrupt -> Invoke hangs.
  - Whether any non-ethos-u CPU operators are left over.

Usage: python vela_tf_show.py -model <model.tflite> [<model2.tflite> ...]
"""
import argparse
from ethosu.vela.tflite.Model import Model


def show(path):
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    model = Model.GetRootAsModel(buf, 0)

    print(f"=== {path} ===")
    print(f"subgraphs: {model.SubgraphsLength()}   operator_codes: {model.OperatorCodesLength()}")
    codes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        custom = oc.CustomCode()
        codes.append(custom.decode() if custom else f"builtin_{oc.BuiltinCode()}")
    print(f"operator types: {codes}")

    for s in range(model.SubgraphsLength()):
        sg = model.Subgraphs(s)
        print(f"\n--- subgraph[{s}] operators={sg.OperatorsLength()} tensors={sg.TensorsLength()} ---")
        print(f"  graph inputs  : {[sg.Inputs(i) for i in range(sg.InputsLength())]}")
        print(f"  graph outputs : {[sg.Outputs(i) for i in range(sg.OutputsLength())]}")

        def tname(idx):
            t = sg.Tensors(idx)
            shape = [t.Shape(k) for k in range(t.ShapeLength())]
            return f"{t.Name().decode()} shape={shape} type={t.Type()} buf={t.Buffer()}"

        for o in range(sg.OperatorsLength()):
            op = sg.Operators(o)
            kind = codes[op.OpcodeIndex()]
            ins = [op.Inputs(i) for i in range(op.InputsLength())]
            outs = [op.Outputs(i) for i in range(op.OutputsLength())]
            print(f"\n  op[{o}] {kind}  ({len(ins)} inputs -> {len(outs)} outputs)")
            for i, idx in enumerate(ins):
                print(f"      in [{i}] #{idx}: {tname(idx)}")
            for i, idx in enumerate(outs):
                print(f"      out[{i}] #{idx}: {tname(idx)}")

        # Summarize every tensor whose name mentions scratch/fast/command/weight
        print("\n  tensors matching scratch/fast/command/weight:")
        for t_i in range(sg.TensorsLength()):
            t = sg.Tensors(t_i)
            n = t.Name().decode()
            if "scratch" in n.lower() or "fast" in n.lower() or "command" in n.lower() or "weight" in n.lower():
                shape = [t.Shape(k) for k in range(t.ShapeLength())]
                nbytes = 1
                for x in shape:
                    nbytes *= max(x, 1)
                print(f"    #{t_i} {n}  shape={shape} (~{nbytes/1024:.1f}KB) type={t.Type()}")


def main():
    ap = argparse.ArgumentParser(
        description="Dump the flatbuffer structure of a Vela-generated tflite model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example: python vela_tf_show.py -model iris/irisnet_lowres_16x8_vela.tflite",
    )
    ap.add_argument(
        "-model", "--model",
        dest="models",
        nargs="+",
        required=True,
        metavar="TFLITE",
        help="one or more Vela-generated .tflite models to inspect",
    )
    args = ap.parse_args()

    for path in args.models:
        show(path)


if __name__ == "__main__":
    main()
