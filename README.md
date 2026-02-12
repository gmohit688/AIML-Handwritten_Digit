# AIML-Handwritten_Digit
# Handwritten Digit Recognition using CNN (TensorFlow)

An end-to-end Deep Learning project that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

This project demonstrates the complete ML workflow — from data preprocessing to model training, evaluation, saving, and real-world inference.

---

## 🚀 Project Overview

This system:

- Loads and preprocesses the MNIST dataset
- Builds a Convolutional Neural Network (CNN)
- Trains and evaluates the model
- Saves the trained model in `.keras` format
- Loads the saved model for prediction (inference)
- Predicts digits from custom handwritten images

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- PIL (Python Imaging Library)

---

## 📂 Project Structure
HandwrittenDigit/
│
├── data/
│ └── sample_images/
│ └── digit.png
│
├── model/
│ ├── init.py
│ ├── cnn_model.py
│ ├── train.py
│ ├── predict.py
│ └── mnist_cnn_model.keras
│
├── venv/
└── README.md

## 🔍 Model Architecture

The CNN architecture includes:

- Convolution layers (feature extraction)
- ReLU activation
- MaxPooling layers
- Flatten layer
- Dense layers
- Softmax output layer (10 classes)

The model outputs probability scores for digits 0–9.

---

## 🏋️ Training the Model

From the project root (with virtual environment activated):

```bash
python model/train.py
This will:

Train the CNN

Evaluate on test data

Save the model as:

model/mnist_cnn_model.keras

🔮 Running Prediction (Inference)

Place a handwritten digit image inside:

data/sample_images/


Then run:

python model/predict.py


Example Output:

Predicted Digit: 5
Confidence: 0.98

⚠️ Important Notes
> Image size must be 28x28
> Grayscale format
> Pixel values normalized (0–1)
> MNIST format: white digit on black background


<img width="667" height="455" alt="image" src="https://github.com/user-attachments/assets/4f70ecc9-a448-4ded-be79-f4a0c19f625f" />

