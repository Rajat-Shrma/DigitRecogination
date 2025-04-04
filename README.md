# Digit Recognition App

A web-based application for recognizing handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Built with Streamlit and TensorFlow.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Model Architecture](#model-architecture)

## About

The **Digit Recognition App** is an interactive tool that allows users to draw a digit on a canvas and receive a prediction of the drawn digit using a pre-trained deep learning model. The project leverages a Convolutional Neural Network (CNN) trained on the MNIST dataset, a standard benchmark for handwritten digit classification. The app is built using Streamlit for the frontend and TensorFlow for the backend model, making it both accessible and powerful.

This project is ideal for learning about image classification, CNNs, and deploying machine learning models in a user-friendly interface.

## Features

- **Interactive Canvas**: Draw digits directly in the browser using a Streamlit drawable canvas.
- **Real-Time Prediction**: Predicts the digit using a pre-trained CNN model with a single click.
- **High Accuracy**: Achieves over 99% accuracy on the MNIST test set.
- **Simple Interface**: Minimalist design with clear instructions and results.

## Model Architecture

The CNN model is implemented using TensorFlow/Keras with the following architecture:

```python

model = Sequential([
    # Convolutional Layer 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    # Convolutional Layer 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Convolutional Layer 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),

    # Flatten the feature maps
    Flatten(),

    # Dense Layer
    Dense(128, activation='relu'),

    # Output Layer
    Dense(10, activation='softmax')
])
