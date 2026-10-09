# 🐱🐶 Cats vs Dogs Classification Using CNN

## Project Overview

This project uses a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** to classify images as either cats or dogs. The project also uses **data augmentation** and **dropout** to help reduce overfitting and improve generalization.

## Dataset

**Source:** [Cats and Dogs Light Dataset — Kaggle](https://www.kaggle.com/datasets/mukeshmanral/cats455-and-dogs545-light-1000)

The dataset contains **1,000 images**: 455 cat images and 545 dog images.

### Dataset Split

First, 80% of the images were allocated to the training portion and 20% to the test set. Then, 10% of the training portion was reserved for validation.

| Subset | Percentage of total | Number of images |
|---|---:|---:|
| Training | 72% | 720 |
| Validation | 8% | 80 |
| Testing | 20% | 200 |

## Libraries

```python
import cv2
from tensorflow.keras import models, layers, utils
import matplotlib.pyplot as plt
```

- **OpenCV:** Image processing
- **TensorFlow/Keras:** Building and training the CNN
- **Matplotlib:** Plotting training metrics

## Image Preprocessing

- Resize images to **128 × 128** pixels.
- Use three color channels (RGB) as model input.
- Normalize pixel values to the **[0, 1]** range using `layers.Rescaling(1/255.0)`.

## CNN Architecture

```python
model_cnn = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1/255.0),
    layers.Conv2D(kernel_size=(3, 3), filters=16, padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(kernel_size=(3, 3), filters=32, padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(units=32, activation="relu"),
    layers.Dense(units=16, activation="relu"),
    layers.Dense(units=8, activation="relu"),
    layers.Dense(units=2, activation="softmax")
])
```

| Layer | Configuration |
|---|---|
| Input | 128 × 128 × 3 |
| Rescaling | 1/255 |
| Conv2D | 16 filters, 3×3, ReLU, same padding |
| MaxPooling2D | 2×2 (default) |
| Conv2D | 32 filters, 3×3, ReLU, same padding |
| MaxPooling2D | 2×2 (default) |
| Flatten | Feature maps to one-dimensional vector |
| Dense | 32 neurons, ReLU |
| Dense | 16 neurons, ReLU |
| Dense | 8 neurons, ReLU |
| Output | 2 neurons, Softmax |

The convolutional layers learn visual features, pooling reduces spatial dimensions, and dense layers generate the final class probabilities.

## Techniques to Improve Generalization

### Data Augmentation

Data augmentation was applied during training to introduce additional image variation and help the model generalize beyond the original training images.

### Dropout

Dropout was used during training to reduce overfitting by randomly disabling a proportion of activations.

> **Note:** The architecture snippet above reproduces the model code provided for this README. The specific augmentation operations, dropout rate, and dropout placement were not supplied, so they are not shown in the code snippet.

## Training Results

The following plot shows accuracy across training epochs:

![Training and Validation Accuracy by Epoch](output1.png)

> Place `output1.png` in the same directory as `README.md` to display it on GitHub.

## Project Objectives

- Build a CNN for binary image classification.
- Practice image preprocessing with OpenCV and Keras.
- Apply data augmentation and dropout to address overfitting.
- Track model learning using accuracy curves.
- Evaluate the model on a held-out test set.

## Conclusion

This project demonstrates a CNN-based workflow for distinguishing cats from dogs. The model combines convolutional feature extraction, pooling, and dense classification layers, with augmentation and dropout used to support generalization.

*Test accuracy and the number of training epochs can be added once the final experiment results are available.*
