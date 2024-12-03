1. **ACE_Implementation.ipynb**
- Implements the Automatic Concept Extraction (ACE) method
- Used to automatically identify and extract concepts from neural networks
- Helps understand what features the model is actually using for classification

2. **Craft Demo.ipynb**
- Demonstrates the usage of CRAFT (Concept Reasoning And Feature Translation)
- Downloads and processes example data (like the rabbit.npz file)
- Shows how to apply concept-based reasoning to understand model decisions

3. **Experiments.ipynb**
- Sets up experimental configurations
- Defines directories and parameters for running tests
- Uses ResNet18 model with specific configurations for the "corgi" class
- Manages experimental parameters like batch size, learning rate, etc.

4. **Training_Pipeline.ipynb**
- Handles the core training process
- Installs necessary dependencies including the custom "spawrious" package
- Sets up the training environment and data loading
- Implements the main training loop for the models

The project is focused on:
1. Training image classifiers
2. Using multiple concept activation methods (ACE and CRAFT)
3. Creating a mathematical framework to automatically detect spurious correlations
4. Analyzing how models make decisions and identifying potential biases

## Spurious Detector

The `spurious_detector.py` module provides tools for detecting and analyzing spurious correlations in neural network models. It works by analyzing the relationship between concept activations, their importances, and model predictions.

### How It Works

The `SpuriousDetector` class operates in three main steps:

1. **Initialization**
   - Loads pre-computed concept vectors and their importances
   - Takes a trained model and validation dataset as input
   - Requires results from previous concept extraction (using ACE or CRAFT)

2. **Spurious Score Calculation**
   - For each concept, calculates:
     - Concept importance from pre-computed data
     - Correlation between concept activations and model predictions
     - Combined score using both metrics

3. **Spurious Concept Identification**
   - Uses a threshold-based approach to identify potentially spurious concepts
   - Ranks concepts by their combined spurious scores
   - Saves analysis results for further investigation

### Usage
