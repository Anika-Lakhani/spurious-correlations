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