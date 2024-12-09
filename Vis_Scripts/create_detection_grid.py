import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import glob
import os

def create_detection_grid(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get unique pairings
    pairings = df['pairing'].unique()
    
    for pairing in pairings:
        print(f"\nProcessing pairing: {pairing}")
        # Filter data for this pairing
        pairing_data = df[df['pairing'] == pairing]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Prepare data for heatmap
        methods = ['med-mean', 'mean', 'median']
        concepts = sorted(pairing_data['num_concepts'].unique())
        
        # Create matrix
        matrix = []
        
        # Function to convert accuracy to text
        def acc_to_text(acc):
            return "buggy model had\nhigher spurious score" if acc == 1.0 else "normal model had\nhigher spurious score"
        
        for method in methods:
            row = []
            for num_concepts in concepts:
                # Get accuracy for models
                model_data = pairing_data[
                    (pairing_data['method'] == method) & 
                    (pairing_data['num_concepts'] == num_concepts)
                ]
                
                # Get accuracy value
                acc = model_data['accuracy'].iloc[0]
                row.append(acc)
            
            matrix.append(row)
        
        # Create heatmap with custom annotations
        ax = plt.gca()
        
        # Create the heatmap for colors
        sns.heatmap(matrix, 
                   cmap=['#ff9999', '#2e8b57'],  # Red for 0, Green for 1
                   vmin=0.0,  # Force the minimum value to be 0.0
                   vmax=1.0,  # Force the maximum value to be 1.0
                   cbar=False,
                   xticklabels=[f'{n} concepts' for n in concepts],
                   yticklabels=methods,
                   annot=False)
        
        # Add custom text annotations
        for i in range(len(methods)):
            for j in range(len(concepts)):
                text = acc_to_text(matrix[i][j])
                ax.text(j + 0.5, i + 0.5, text,
                       ha='center', va='center',
                       fontsize=8,
                       color='black',
                       wrap=True)
        
        # Customize plot
        plt.title(f'Detection Results for {pairing}', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Concepts')
        plt.ylabel('Method')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_dir = 'final_results/o2o_easy/o2o_easy_graphs/detection_grids'
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = f'{pairing}_detection_grid.png'
        output_path = os.path.join(output_dir, plot_filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py path/to/detection_accuracies.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    create_detection_grid(csv_path)