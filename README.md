# CIFAR CNN Models



## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

## Introduction

This repository contains two Convolutional Neural Network (CNN) models trained from scratch for image classification on the CIFAR-10 and CIFAR-100 datasets. The models are designed to classify images into 10 and 100 categories, respectively.

## Features

- Trained CNN models for CIFAR-10 and CIFAR-100.
- Data preprocessing and augmentation.
- Model evaluation with accuracy and loss metrics.
- Confusion matrices and classification reports.
- Scripts for loading models and making predictions.

## Technologies Used

- **Python 3.x**
- **TensorFlow 2.x / Keras**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Git & GitHub**

## Dataset

### CIFAR-10

- **Description:** 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
- **Train/Test Split:** 50,000 training images and 10,000 test images.

### CIFAR-100

- **Description:** 60,000 32x32 color images in 100 classes, with 600 images per class.
- **Classes:** 100 fine-grained classes grouped into 20 superclasses.
- **Train/Test Split:** 50,000 training images and 10,000 test images.

## Model Architecture

### Common Architecture for Both Models

1. **Convolutional Layers:**
   - Multiple Conv2D layers with ReLU activation.
   - Batch Normalization after each Conv2D layer.
2. **Pooling Layers:**
   - MaxPooling2D to reduce spatial dimensions.
3. **Dropout Layers:**
   - Dropout to prevent overfitting.
4. **Fully Connected Layers:**
   - Dense layers with ReLU activation.
   - Batch Normalization and Dropout.
5. **Output Layer:**
   - Dense layer with Softmax activation for classification.

### CIFAR-10 Model

- **Output Units:** 10 (corresponding to CIFAR-10 classes).

### CIFAR-100 Model

- **Output Units:** 100 (corresponding to CIFAR-100 classes).

## Training

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Data Augmentation:** Rotation, shifting, flipping, zooming.
- **Callbacks:**
  - Early Stopping
  - Model Checkpointing

## Evaluation

- **Test Accuracy:**
  - CIFAR-10: 85%
  - CIFAR-100: 55%
- **Confusion Matrix**
- **Classification Report**

## Usage

### Loading the Models

python
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load CIFAR-10 model
cnn_c10 = load_model('models/cnn_cifar10.h5')

# Load CIFAR-100 model
cnn_c100 = load_model('models/cnn_cifar100.h5')

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.

## Contact
Name: Anish vijayvergiya
Email: anishvijayvergiya1010.com
LinkedIn: Anish vijay
GitHub: Anishvjy
