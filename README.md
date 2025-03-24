# Handwritten-Digit-Classifier
This is a simple handwritten digit classifier which uses Convolutional Neural Network to classify the input image and predict the digit present in the image.

---

This project implements a simple Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The trained model is saved as `mark1.keras`.

## Features

- Uses a CNN model for handwritten digit recognition.
- Trained on the MNIST dataset with 10 epochs.
- Model is saved as `mark1.keras` for inference.
- Includes a script to classify a given handwritten digit image.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/handwritten-digit-classification.git
   cd handwritten-digit-classification
   ```

2. Install dependencies:
   ```sh
   pip install tensorflow numpy matplotlib pandas scikit-learn
   ```

## Training the Model

To train the model, run:
```sh
python digit_classification.py
```
This will train the CNN on the MNIST dataset and save the model as `mark1.keras`.

## Using the Model for Prediction

To classify a handwritten digit image, run:
```sh
python digit_class.py
```
Then enter the path to the image when prompted.

## Model Architecture

- **3 Convolutional Layers** (with ReLU activation and max pooling)
- **Flatten Layer**
- **Fully Connected Dense Layers**
- **Softmax Output Layer** (for 10-class classification)

## Example Usage

If you have an image `digit.png` containing a handwritten digit, you can classify it as follows:
```sh
python digit_class.py
```
Then, enter:
```
path/to/digit.png
```
The script will output the predicted digit.

## License

This project is licensed under the MIT License.

---
