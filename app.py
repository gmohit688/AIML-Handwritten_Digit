from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io

app = Flask(__name__)

model = tf.keras.models.load_model("model/mnist_cnn_model.keras")


def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32")

    if np.mean(image_array) > 127:
        image_array = 255 - image_array

    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        img_data = request.form["image"]
        img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data)

        image = Image.open(io.BytesIO(img_bytes))
        processed = preprocess_image(image)

        predictions = model.predict(processed)
        prediction = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
