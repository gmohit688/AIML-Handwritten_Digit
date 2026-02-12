import tensorflow as tf

from model.cnn_model import build_cnn_model
from load_data import load_and_prepare_mnist


def train_model():
    """
    Compiles, trains, evaluates, and saves the CNN model.
    """

    # Load prepared data
    x_train, y_train, x_test, y_test = load_and_prepare_mnist()

    # Build CNN model
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=10)

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1
    )

    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save trained model
    model.save("model/mnist_cnn_model.keras")
    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()
