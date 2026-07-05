import os
import shutil

import numpy as np
import tensorflow as tf

try:
    import onnx
    import onnxruntime
except ImportError as exc:
    raise SystemExit("onnx and onnxruntime are required for export/inference") from exc


onnx_model_file = "onnx/ops_softmax.onnx"
pb_model_file = "pb/ops_softmax.pb"
os.makedirs(os.path.dirname(onnx_model_file), exist_ok=True)
os.makedirs(os.path.dirname(pb_model_file), exist_ok=True)


class SoftmaxModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x):
        return self.softmax(x)


def export_to_pb():
    model = SoftmaxModel()
    model.build(input_shape=(None, 1, 5, 10))

    export_dir = os.path.join("pb", "ops_softmax")
    input_spec = tf.TensorSpec(shape=(1, 1, 5, 10), dtype=tf.float32, name="input")
    signatures = model.call.get_concrete_function(input_spec)

    tf.saved_model.save(model, export_dir, signatures=signatures)

    saved_model_pb = os.path.join(export_dir, "saved_model.pb")
    shutil.copyfile(saved_model_pb, pb_model_file)
    print(f"Export PB: {pb_model_file}")


def export_to_onnx():
    model = SoftmaxModel()
    print(model)

    model.build(input_shape=(None, 1, 5, 10))
    x = tf.range(50.0, dtype=tf.float32)
    x = tf.reshape(x, (1, 1, 5, 10))

    outs = model(x)
    print("TensorFlow out:", outs.numpy())

    try:
        import tf2onnx
    except ImportError as exc:
        raise SystemExit("tf2onnx is required to export to ONNX") from exc

    spec = (tf.TensorSpec((1, 1, 5, 10), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open(onnx_model_file, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"Export ONNX: {onnx_model_file}")


def infer_onnx():
    onnx_model = onnx.load(onnx_model_file)
    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())

    x = np.arange(50.0, dtype=np.float32).reshape(1, 1, 5, 10)
    input_fd = {"input": x}
    outs = sess.run(None, input_fd)

    for arr, t in zip(outs, sess.get_outputs()):
        print("output name:", t.name, ", data:", arr)


if __name__ == "__main__":
    export_to_pb()
    export_to_onnx()
    infer_onnx()
