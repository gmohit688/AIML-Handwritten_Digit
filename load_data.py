import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_and_prepare_mnist():
    """
    Loads the MNIST dataset, normalizes pixel values,
    and reshapes data for CNN input.
    """
    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape to add channel dimension (CNN requirement)
    # From: (samples, 28, 28)
    # To:   (samples, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_and_prepare_mnist()

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Visualize one image
    plt.imshow(x_train[0].squeeze(), cmap="gray")
    plt.title(f"Label: {y_train[0]}")
    plt.axis("off")
    plt.show()
