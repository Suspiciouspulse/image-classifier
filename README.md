🐱 vs 🐶 Cats vs Dogs Image Classification (TensorFlow)

A Convolutional Neural Network (CNN) implementation capable of classifying images of cats and dogs with high accuracy. This project was built as part of the freeCodeCamp Machine Learning curriculum to demonstrate an end-to-end deep learning pipeline for computer vision.
📖 Overview

This project implements a binary image classifier using TensorFlow and Keras. It takes raw images as input, processes them through a deep learning pipeline, and outputs a probability indicating whether the image contains a cat or a dog.

Key objectives achieved:

    Applied image preprocessing and data augmentation to improve model generalization.

    Designed and trained a custom CNN architecture.

    Evaluated performance using unseen test data.

🚀 Key Features

    Data Preprocessing: Automated image resizing and pixel normalization (scaling pixel values to 0-1 range).

    Data Augmentation: Implemented ImageDataGenerator to perform random rotations, shifts, flips, and zooms during training to prevent overfitting.

    CNN Architecture: A custom Sequential model utilizing:

        Convolutional Layers (Conv2D): For feature extraction.

        Pooling Layers (MaxPooling2D): for spatial down-sampling.

        Dropout: To reduce overfitting.

        Dense Layers: For classification.

    Binary Classification: Uses a Sigmoid activation function and Binary Cross-Entropy loss to determine class probabilities.

🛠️ Technologies Used

    Python: Primary programming language.

    TensorFlow / Keras: Deep learning framework for building and training the model.

    NumPy: For numerical operations and array handling.

    Matplotlib: For visualizing training results (accuracy/loss graphs) and image predictions.

🧠 Model Architecture

The model is a Sequential Convolutional Neural Network. It creates a map of features by sliding filters across the input image.

Shutterstock

    Input Layer: Accepts RGB images (formatted to specific dimensions, e.g., 150x150).

    Feature Extraction: Multiple blocks of Conv2D + MaxPooling2D layers.

    Flattening: Converts 2D feature maps into a 1D vector.

    Classification: Fully connected Dense layers with Dropout regularization.

    Output: A single neuron with sigmoid activation.

📂 Dataset

The dataset was provided by freeCodeCamp. It consists of a labeled collection of images split into three subsets:

    Training Set: Used to teach the model.

    Validation Set: Used to tune hyperparameters and monitor training progress.

    Test Set: Unseen data used for the final performance evaluation.

💻 Getting Started

To run this project locally:

    Clone the repository:
    Bash

git clone https://github.com/your-username/cats-vs-dogs-cnn.git

Install dependencies:
Bash

    pip install tensorflow numpy matplotlib

    Run the notebook/script: Open the Jupyter Notebook or run the Python script to train the model.

🤝 Acknowledgments

    freeCodeCamp: For the curriculum and dataset provision.

    TensorFlow Team: For the excellent documentation and tools.
