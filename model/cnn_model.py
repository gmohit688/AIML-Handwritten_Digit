import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds and returns a CNN model for handwritten digit recognition.
    """

    model = models.Sequential(name="mnist_cnn")

    # -------- Convolution Block 1 --------
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=input_shape
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # -------- Convolution Block 2 --------
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # -------- Fully Connected Layers --------
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
