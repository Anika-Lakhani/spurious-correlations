import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def create_activation_plot(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Number of concepts
    n_concepts = len(df)
    
    # Set up the plot with extra bottom margin
    plt.figure(figsize=(12, 8))
    
    # Width of each bar
    bar_width = 0.35
    
    # Positions for bars
    indices = range(n_concepts)
    
    # Create bars
    plt.bar([i - bar_width/2 for i in indices], 
            df['class_activation'] * 100,
            bar_width, 
            label='Class Activation',
            color='skyblue')
    
    plt.bar([i + bar_width/2 for i in indices], 
            df['background_activation'] * 100,
            bar_width, 
            label='Background Activation',
            color='lightcoral')
    
    # Customize the plot
    plt.xlabel('Concept')
    plt.ylabel('Activation (%)')
    plt.title('Class vs Background Activations by Concept')
    plt.legend()
    
    # Create two-line labels: concept number/letter and ratio
    if 'normal' in csv_path.lower():
        labels = [f'Concept {i+1}\nRatio: {df.iloc[i]["ratio"]:.2f}' for i in indices]
    else:
        # Convert numbers to letters (A, B, C, etc)
        labels = [f'Concept {chr(65+i)}\nRatio: {df.iloc[i]["ratio"]:.2f}' for i in indices]
    plt.xticks(indices, labels)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add extra bottom margin
    plt.subplots_adjust(bottom=0.6)
    
    # Save the plot using the CSV filename (without extension) as the plot filename
    plot_filename = csv_path.rsplit('.', 1)[0] + '_plot.png'
    # Create the output directory path
    output_dir = 'final_results/o2o_easy/o2o_easy_graphs/activation_plots'
    os.makedirs(output_dir, exist_ok=True)
    # Save to the new location
    output_path = os.path.join(output_dir, os.path.basename(plot_filename))
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py path/to/csv_file")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    create_activation_plot(csv_path)