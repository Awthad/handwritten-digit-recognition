# Handwritten Digit Recognition

An end-to-end machine learning project for recognizing handwritten digits using the MNIST dataset. The project compares multiple machine learning and deep learning approaches and deploys the final CNN model as a web application.

## Project Overview

The goal of this project was to build and evaluate different models for handwritten digit classification, then deploy the best-performing model.

Models tested:

* Logistic Regression
* Random Forest
* Dense Neural Network
* Convolutional Neural Network (CNN)

## Results

| Model                | Test Accuracy |
| -------------------- | ------------: |
| Logistic Regression  |        92.64% |
| Random Forest        |        97.04% |
| Dense Neural Network |        97.74% |
| CNN                  |    **99.10%** |

The CNN performed best, achieving **99.10% test accuracy** with only 90 incorrect predictions out of 10,000 test images.

## Tech Stack

* Python
* NumPy
* scikit-learn
* TensorFlow / Keras
* Pillow
* Flask
* Jupyter Notebook

## Features

* MNIST handwritten digit classification
* Comparison of traditional machine learning and neural network models
* CNN-based image recognition
* Saved trained model using Keras
* Flask web application

## Project Structure

```text
handwritten-digit-recognition/
├── app.py
├── web_app.py
├── mnist_cnn.keras
├── requirements.txt
├── .gitignore
└── mist_training.ipynb
```

## Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the web application:

```bash
python web_app.py
```

Then open:

```text
http://localhost:5000
```


## Future Improvements

* Add an interactive browser drawing canvas
* Improve preprocessing for user-drawn digits
* Add confidence visualization
* Experiment with CNN architecture and data augmentation
* Deploy the application online

## Author

Aw Tha Da (Hendrick)
