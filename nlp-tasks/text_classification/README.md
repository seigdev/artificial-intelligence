# Scam URL Classification Project

## Overview

This project focuses on classifying URLs as either legitimate (0) or scam (1) using machine learning techniques. The approach combines feature engineering with different embedding methods to create robust models for scam detection.

## Key Features

- Data Analysis: Explores URL characteristics like length, domain age, and suspicious keywords
- Feature Engineering: Extracts meaningful features from URLs including:
  - URL length
  - Presence of special characters
  - IP addresses in URLs
  - Suspicious keywords
  - Typo-squatting detection
  - Domain age
  - URL shortening services
- Embedding Methods: Utilizes three different embedding techniques:
  - Custom-trained Word2Vec (300D)
  - Google's NNLM (50D)
  - Google's NNLM (128D)
- Model Architecture: Implements a Multi-Layer Perceptron (MLP) neural network with:

  - Input layer matching embedding dimensions
  - Two hidden layers with ReLU activation
  - Dropout layers for regularization
  - Sigmoid output for binary classification

## Results

The project evaluates three different models combining engineered features with each embedding method:

1. Word2Vec + Features Model
   - Accuracy: **87.99%**
   - Confusion matrix visualization
   - Training/validation accuracy and loss plots
2. NNLM-50D + Features Model
   - Accuracy: **87.72%**
   - Confusion matrix visualization
   - Training/validation accuracy and loss plots
3. NNLM-128D + Features Model
   - Accuracy: **88.08%**
   - Confusion matrix visualization
   - Training/validation accuracy and loss plots

## Installation

To run this project, you'll need to install the necessary libraries:

`pip install numpy==1.26.0 tensorflow==2.18.0\
pip install pandas gensim tensorflow_hub matplotlib seaborn tldextract whois rapidfuzz`

## Usage

1. Load the notebook in Google Colab or Jupyter
2. Run cells sequentially to:
   - Load and preprocess data
   - Perform exploratory data analysis
   - Extract features
   - Train and evaluate models
