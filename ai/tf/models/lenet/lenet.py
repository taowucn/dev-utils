import argparse
import os
import sys

import numpy as np
import tensorflow as tf

BATCH_SIZE = 64
WEIGHTS_FILE = "lenet_weights.h5"
MODEL_FILE = "lenet_model.pb"

class LeNet5(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, kernel_size=5, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(16, kernel_size=5, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        ])
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation="relu"),
            tf.keras.layers.Dense(84, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

    def call(self, x):
        x = self.conv(x)
        return self.fc(x)


def build_model():
    model = LeNet5()
    model.build(input_shape=(None, 28, 28, 1))
    return model


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    return x_train, y_train, x_test, y_test


def run_train(args):
    x_train, y_train, x_test, y_test = load_mnist_data()

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training LeNet-5 on MNIST...")
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=1,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    model.save_model(MODEL_FILE)
    model.save_weights(WEIGHTS_FILE)
    print(f"Saved TensorFlow model weights to: {WEIGHTS_FILE}")


def run_test(args):
    x_train, y_train, x_test, y_test = load_mnist_data()

    model = build_model()
    model.load_weights(WEIGHTS_FILE)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    if args.i:
        print("Input image path:", args.i)
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image inference") from exc

        image = Image.open(args.i).convert("L")
        image = image.resize((28, 28))
        image_array = np.asarray(image, dtype="float32") / 255.0
        image_array = image_array[..., None]
        image_array = np.expand_dims(image_array, axis=0)

        preds = model.predict(image_array, verbose=0)[0]
        predicted = int(np.argmax(preds))
        print(f"Predicted label: {predicted}")
        print("Scores:", preds)


def main(args):
    if args.train:
        run_train(args)
    elif args.test:
        run_test(args)
    else:
        print("No action")


def init_param(argv):
    parser = argparse.ArgumentParser(description="Train and test LeNet with TensorFlow and MNIST")
    parser.add_argument("-train", action="store_true", required=False, help="Train model")
    parser.add_argument("-test", action="store_true", required=False, help="Test model")
    parser.add_argument("-i", type=str, required=False, help="input image file for inference")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = init_param(sys.argv[1:])
    main(args)
