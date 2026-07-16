#!/usr/bin/env python3

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import argparse

from google.protobuf import text_format

try:
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.protobuf import meta_graph_pb2
except Exception:
    from tensorflow.core.framework import graph_pb2, meta_graph_pb2

try:
    from tensorflow.python.framework import tensor_util
except Exception as e:
    tensor_util = None
    TENSOR_UTIL_IMPORT_ERROR = e
else:
    TENSOR_UTIL_IMPORT_ERROR = None


def shape2tuple(shape):
    if shape is None:
        return ()
    return tuple(getattr(d, 'size', 0) for d in shape.dim)


def get_node_op_type(node):
    return getattr(node, 'op', getattr(node, 'op_type', None))


def load_pb_model(model_path):
    model_path = os.path.abspath(model_path)

    if os.path.isdir(model_path):
        saved_model_pb = os.path.join(model_path, "saved_model.pb")
        if not os.path.exists(saved_model_pb):
            raise FileNotFoundError(f"No saved_model.pb found in directory: {model_path}")
        with open(saved_model_pb, "rb") as f:
            payload = f.read()

        meta_graph = meta_graph_pb2.MetaGraphDef()
        meta_graph.ParseFromString(payload)
        return meta_graph, "saved_model"

    with open(model_path, "rb") as f:
        payload = f.read()

    try:
        graph = graph_pb2.GraphDef()
        graph.ParseFromString(payload)
        return graph, "graphdef"
    except Exception:
        meta_graph = meta_graph_pb2.MetaGraphDef()
        meta_graph.ParseFromString(payload)
        return meta_graph, "metagraph"


def show_model(args):
    print("Model:", args.model)
    model_obj, model_type = load_pb_model(args.model)
    print("model_type:", model_type)

    if model_type == "saved_model":
        meta_graph = model_obj
        graph = meta_graph.graph_def
        print("Model type: SavedModel")
        print("Signature keys:", list(meta_graph.signature_def.keys()))
    elif model_type == "metagraph":
        meta_graph = model_obj
        graph = meta_graph.graph_def
        print("Model type: MetaGraphDef")
        print("Signature keys:", list(meta_graph.signature_def.keys()))
    else:
        graph = model_obj
        print("Model type: GraphDef")

    node = graph.node
    print("Node num:", len(node))

    if args.d == 1:
        for i in range(len(node)):
            print("name:", node[i].name, " op_type:", get_node_op_type(node[i]))
    elif args.d == 2:
        print(text_format.MessageToString(graph))

    if args.exportw:
        if tensor_util is None:
            raise RuntimeError(
                "tensor_util is unavailable in this TensorFlow environment; "
                "install a compatible TensorFlow/protobuf combination to export weights."
            ) from TENSOR_UTIL_IMPORT_ERROR

        out_folder = args.exportw
        os.makedirs(out_folder, exist_ok=True)
        print("\n------ Model Export Weights  ------")
        print("export-weights out folder:", out_folder)

        for i, n in enumerate(node):
            if get_node_op_type(n) != "Const":
                continue
            if "value" not in n.attr:
                continue
            value_proto = n.attr["value"].tensor
            arr = tensor_util.MakeNdarray(value_proto)
            if args.f:
                arr = arr.astype(args.f)
            out_fn = os.path.join(out_folder, f"{n.name}.bin")
            arr.tofile(out_fn)
            print("export const weight:", n.name, " -> ", out_fn)

    print("\n------ Model Primary IO  ------")
    print("Input num:", len(graph.node))
    print("Output num:", len(graph.node))

    input_nodes = [n.name for n in graph.node if get_node_op_type(n) == "Placeholder"]
    output_nodes = [n.name for n in graph.node if get_node_op_type(n) == "Identity"]
    if not output_nodes:
        output_nodes = [
            n.name for n in graph.node
            if get_node_op_type(n) not in ("Const", "Placeholder")
        ]

    print("Input names:", input_nodes)
    print("Output names:", output_nodes)


def init_param(args):
    parser = argparse.ArgumentParser(description="Show TensorFlow Protobuf Model")

    parser.add_argument("-model", type=str, required=True, default="model.pb",
        help=".pb or saved_model directory")

    parser.add_argument("-d", type=int, required=False, default=0,
        help="Show log level info. [0, 2]")

    parser.add_argument("-exportw", type=str, required=False,
        help="export constant weights to output folder")

    parser.add_argument("-f", type=str, required=False,
        help="export weights with data format, e.g. float32")

    return parser.parse_args(args)


if __name__ == '__main__':
    args = init_param(sys.argv[1:])
    show_model(args)
