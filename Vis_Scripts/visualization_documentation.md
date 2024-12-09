# Visualization Scripts Documentation

## Overview
This documentation covers three visualization scripts for analyzing model behavior and concept activations:
1. `create_activation_plot.py` - Compares class vs background activations for concepts
2. `create_detection_grid.py` - Shows detection results between normal and buggy models
3. `create_spurious_score_plot.py` - Compares spurious scores across different methods and models

## Usage

### 1. Activation Plot (`create_activation_plot.py`)
Creates a bar chart comparing class activation vs background activation for each concept.

```bash
python Vis_Scripts/create_activation_plot.py path/to/activations.csv
python Vis_Scripts/create_activation_plot.py "Final_Results/o2o_easy/bulldog
_desert_bugged_4_concepts.csv"
```

**Input CSV Format:**
```csv
concept,class_activation,background_activation,ratio
1,0.75,0.25,3.0
2,0.65,0.30,2.17
...
```

**Output:** Bar chart showing:
- Blue bars: Class activation percentages
- Red bars: Background activation percentages
- X-axis: Concepts (numbered 1,2,3... or lettered A,B,C... based on filename)
- Each concept shows its activation ratio below

### 2. Detection Grid (`create_detection_grid.py`)
Creates a heatmap grid showing detection results between normal and buggy models.

```bash
python Vis_Scripts/create_detection_grid.py path/to/detection_accuracies.csv
python Vis_Scripts/create_detection_grid.py "Final_Results/o2o_easy/detection_accuracies_easy.csv"
```

**Input CSV Format:**
```csv
pairing,method,num_concepts,accuracy
model1_vs_model2,med-mean,3,1.0
model1_vs_model2,mean,3,0.0
...
```

**Output:** Heatmap grid showing:
- Rows: Different methods (med-mean, mean, median)
- Columns: Number of concepts used
- Colors: Green (buggy model had higher score) vs Red (normal model had higher score)
- Separate grid for each model pairing

### 3. Spurious Score Plot (`create_spurious_score_plot.py`)
Creates a grouped bar chart comparing spurious scores across methods and models.

```bash
python Vis_Scripts/create_spurious_score_plot.py path/to/spurious_scores.csv
```

**Input CSV Format:**
```csv
method_model,num_concepts,spurious_score
mean_normal,3,0.85
mean_bugged,3,0.92
...
```

**Output:** Grouped bar chart showing:
- Green bars: Normal model scores
- Red bars: Buggy model scores
- Grouped by number of concepts
- Sub-grouped by method (mean, median, med-mean)
- Labels inside bars when space permits

## Example Workflow

1. Generate activation comparisons:
```bash
python Vis_Scripts/create_activation_plot.py results/normal_activations.csv
python Vis_Scripts/create_activation_plot.py results/buggy_activations.csv
```

2. Compare model detection results:
```bash
python Vis_Scripts/create_detection_grid.py results/detection_results.csv
```

3. Analyze spurious scores:
```bash
python Vis_Scripts/create_spurious_score_plot.py results/spurious_scores.csv
```

## Notes
- All scripts save output plots as PNG files with 300 DPI
- Output filenames are based on input CSV names
- All plots include appropriate titles, labels, and legends
- Grid lines are included for better readability
- Colors are chosen for accessibility and clarity
