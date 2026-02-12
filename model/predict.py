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
    
    image_array = np.array(image).astype("float32")
    # Auto-invert if background is white
    if np.mean(image_array) > 127:
        image_array = 255 - image_array
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
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
    image_path = "data/sample_images/digit2.png"
    digit, confidence = predict_digit(image_path)

    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence:.2f}")
