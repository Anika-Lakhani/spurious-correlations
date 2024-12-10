# CODEMIC: Concept Detection Method for Identifying Correlations

CODEMIC is a novel method for detecting spurious correlations in image classification models using concept activations. This project implements CODEMIC to analyze and compare normal and bugged image classification models trained on a dataset of dogs in various backgrounds.

## Overview

The CODEMIC method uses concept-based explanations to identify when a model has learned spurious correlations between irrelevant background features and target classes. It analyzes concept activations and importances to quantify how much a model relies on background concepts vs. class-specific concepts.

Key components:

- Dataset generation using stable diffusion
- Training of normal and bugged ResNet18 models 
- Extraction of concepts using CRAFT (Concept Recursive Activation FacTorization)
- Calculation of concept activations and importances
- Computation of spurious scores to detect bugged models

## Setup

1. Clone this repository
2. Install required dependencies:

```
pip install -r requirements.txt
```

3. Download one of our models or use your own model. Then, run Craft_Implementation.ipynb to extract concepts.

## Usage

The main experiment script is `Testing_pipeline.ipynb`. To run, set up the ipynb file and run the cells in order on your desired model.

This will:
1. Load the selected model
2. Extract concepts and compute activations
3. Calculate spurious scores 
4. Generate visualizations of results

## Key Files

- `Testing_pipeline.ipynb`: Main script to run CODEMIC analysis
- `models/`: Contains ResNet18 models for use in the pipeline
- `datasets/`: Dataset of dogs in various backgrounds (not included in repo due to limited storage space on our computers, but can be generated using stable diffusion with Spawrious generation code; is also included in this Google Drive folder: https://drive.google.com/drive/folders/1uXUz2fQ35pgp3d9Cowi3eRZEoaBZhO4t?usp=drive_link)
- `final_results/`: Saved concept results and visualizations

## Results

When tested in our research, CODEMIC successfully detects the bugged model with 91.67% accuracy using the mean activation aggregation method across different numbers of concepts. Visualizations show the bugged model relies more heavily on background concepts compared to the normal model.

## Future Work

- Conduct user study on interpretability of visualizations
- Extend to other model architectures and datasets
- Develop techniques to automatically mitigate detected spurious correlations

## Citation

If you use this code, please cite:

[Citation information]

Citations:
https://github.com/Anika-Lakhani/spurious-correlations