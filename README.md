# CNN Model for MNIST Classification

This repository contains a Convolutional Neural Network (CNN) implemented in PyTorch for classifying the MNIST dataset. The model utilizes 3 convolutional layers, dropout regularization, and a fully connected layer to output predictions for the 10 classes of digits (0-9).

## Overview

The CNN model is designed for digit classification using the MNIST dataset. It consists of:

    Three convolutional layers, each followed by max-pooling and dropout.
    A fully connected layer with ReLU activation.
    An output layer with softmax activation for classification.

The model is trained using the Adam optimizer and evaluated using the accuracy metric on the test set.

## Model Architecture

The CNN model consists of the following layers:

    Convolutional Layer 1:
        Input channels: 1 (grayscale image)
        Output channels: 32
        Kernel size: 3x3
        Padding: 1

    MaxPooling Layer 1:
        Pooling size: 2x2

    Convolutional Layer 2:
        Input channels: 32
        Output channels: 64
        Kernel size: 3x3
        Padding: 1

    MaxPooling Layer 2:
        Pooling size: 2x2

    Convolutional Layer 3:
        Input channels: 64
        Output channels: 128
        Kernel size: 3x3
        Padding: 1

    MaxPooling Layer 3:
        Pooling size: 2x2

    Fully Connected Layer:
        Input size: Flattened size of the last convolutional layer (128 * 3 * 3)
        Output size: 128 (hidden units)

    Output Layer:
        Output size: 10 (number of classes for digit classification)

The model also uses dropout after each convolutional and fully connected layer for regularization.

## Training and Testing

### Training

The training process uses:

    Loss function: Cross-Entropy Loss
    Optimizer: Adam optimizer with a learning rate of 0.001
    Mini-batch size: 100
    Epochs: 1 (can be adjusted for further training)

The training function logs the loss after each mini-batch.

### Testing

The testing function evaluates the model on the test dataset and calculates the accuracy. The predicted labels for the test data are returned for further analysis.

## Example Output

The script prints the following information during training and testing:

![image](https://github.com/user-attachments/assets/0c9fcf69-5ea1-42ff-8b0a-3c0777b72545)

