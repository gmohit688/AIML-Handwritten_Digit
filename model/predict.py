import tensorflow as tf
import numpy as np
from PIL import Image


MODEL_PATH = "model/mnist_cnn_model.keras"


def preprocess_image(image_path):
    """
    Loads and preprocesses an image for MNIST CNN prediction.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # (28, 28, 1)
    image_array = np.expand_dims(image_array, axis=0)   # (1, 28, 28, 1)

    return image_array


def predict_digit(image_path):
    """
    Loads trained model and predicts digit from image.
    """
    model = tf.keras.models.load_model(MODEL_PATH)

    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)

    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)

    return predicted_digit, confidence


if __name__ == "__main__":
    image_path = "data/sample_images/digit.png"
    digit, confidence = predict_digit(image_path)

    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence:.2f}")
