# adversarial-noise-generator
A Python library for generating adversarial noise on 2D images.

This repository implements a convolutional neural network (CNN) classifier for the MNIST dataset, focusing on adversarial robustness. The model is built using PyTorch and employs a variety of techniques, including data augmentation, adversarial training, and performance tracking using TensorBoard.

# Overview
The goal of this project is to classify MNIST images using a VGG16 architecture, while also evaluating the model’s robustness to untargeted adversarial attacks. The main steps include:
- Preprocessing the MNIST dataset.
- Training a CNN model.
- Generating untargeted adversarial examples to evaluate the model's robustness.
- Logging training and validation performance using TensorBoard.
- Saving the best-performing model based on validation accuracy.

# Requirements
torch (PyTorch)
torchvision
numpy
matplotlib
tensorboard

# Installation
1- Clone the repository:
2- Install the dependencies:
    pip install -r requirements.txt
3- Set up your environment (if using AWS SageMaker, or local setup):
    Ensure CUDA is available for GPU acceleration if you're training on a GPU.

# Data Pipeline
The MNIST dataset is loaded and processed using custom dataset classes. 

# Model Architecture
The model is based on a CNN architecture and uses VGG16 as a backbone for feature extraction. Key parameters include:
Input channels: 1 (grayscale images).
Output classes: 10 (for MNIST digits 0–9).
The model is trained using the Adam optimizer, with a learning rate of 1e-4 and weight decay of 1e-4. Cross-entropy loss is used for classification.

# Training
To start training the model, run the following command:
python train.py

# Results
The best model is saved as best_model.pth in the results directory based on validation accuracy. You can visualize the training progress and metrics in TensorBoard:
tensorboard --logdir=results
