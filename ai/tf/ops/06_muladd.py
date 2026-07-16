import os
import shutil

import numpy as np
import tensorflow as tf

# try:
#     import onnx
#     import onnxruntime
# except ImportError as exc:
#     raise SystemExit("onnx and onnxruntime are required for export/inference") from exc


onnx_model_file = "onnx/ops_muladd.onnx"
pb_model_file = "pb/ops_muladd.pb"
frozen_pb_model_file = "pb/frozen_ops_muladd.pb"
os.makedirs(os.path.dirname(onnx_model_file), exist_ok=True)
os.makedirs(os.path.dirname(pb_model_file), exist_ok=True)
os.makedirs(os.path.dirname(frozen_pb_model_file), exist_ok=True)


class MulAddModel(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 1, 1, 16), dtype=tf.float32, name="input0")])
    def __call__(self, x):
        o = tf.multiply(x, 2)
        return tf.add(o, 1)


def export_to_pb():
    model = MulAddModel()
    export_dir = os.path.join("pb", "ops_muladd")

    tf.saved_model.save(model, export_dir, signatures={"serving_default": model.__call__})

    saved_model_pb = os.path.join(export_dir, "saved_model.pb")
    shutil.copyfile(saved_model_pb, pb_model_file)
    print(f"Export PB: {pb_model_file}")


def export_to_frozen_pb():
    tf.compat.v1.disable_eager_execution()

    graph = tf.Graph()
    with graph.as_default():
        x = tf.compat.v1.placeholder(tf.float32, shape=[1, 1, 1, 16], name="input0")
        y = tf.multiply(x, 2, name="mul")
        z = tf.add(y, 1, name="add")
        tf.identity(z, name="output")

    with tf.compat.v1.Session(graph=graph) as sess:
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            output_node_names=["output"],
        )

    with open(frozen_pb_model_file, "wb") as f:
        f.write(frozen_graph.SerializeToString())

    print(f"Export Frozen PB: {frozen_pb_model_file}")

'''
def export_to_onnx():
    model = MulAddModel()
    print(model)

    x = tf.range(16, dtype=tf.float32)
    x = tf.reshape(x, (1, 1, 1, 16))

    outs = model(x)
    print("TensorFlow out:", outs.numpy())

    try:
        import tf2onnx
    except ImportError:
        print("Skip ONNX export: tf2onnx is not installed in this environment")
        return False

    spec = (tf.TensorSpec((1, 1, 1, 16), tf.float32, name="input0"),)
    model_proto, _ = tf2onnx.convert.from_tf_module(model, input_signature=spec, opset=13)
    with open(onnx_model_file, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"Export ONNX: {onnx_model_file}")
    return True


def infer_onnx():
    if not os.path.exists(onnx_model_file):
        print(f"Skip ONNX inference: {onnx_model_file} was not created")
        return

    onnx_model = onnx.load(onnx_model_file)
    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())

    x = np.arange(16, dtype=np.float32).reshape(1, 1, 1, 16)
    input_fd = {"input0": x}
    outs = sess.run(None, input_fd)

    for arr, t in zip(outs, sess.get_outputs()):
        print("output name:", t.name, ", data:", arr)
'''

if __name__ == "__main__":
    export_to_pb()
    export_to_frozen_pb()
    # export_to_onnx()
    # infer_onnx()
