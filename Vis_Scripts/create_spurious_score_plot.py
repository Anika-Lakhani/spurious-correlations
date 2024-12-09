import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def create_spurious_score_plot(csv_path):
    # Read the CSV file
    print(f"Attempting to read: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Split method_model into separate columns
    df[['method', 'model_type']] = df['method_model'].str.split('_', expand=True)
    
    # Set up the plot
    plt.figure(figsize=(15, 8))
    
    # Width of each bar
    bar_width = 0.12
    # Gap between normal and bugged groups
    gap = 0.2
    
    # Get unique numbers of concepts
    concept_nums = sorted(df['num_concepts'].unique())
    methods = ['mean', 'median', 'med-mean']
    
    # Colors
    normal_color = '#2e8b57'  # Darker sea green
    normal_edge = '#1a543a'   # Even darker green for outline
    bugged_color = '#ff9999'  # Pastel red
    bugged_edge = '#cc3333'   # Darker red for outline
    
    # Create bars for each group
    for i, num_concepts in enumerate(concept_nums):
        # First create all normal bars
        for j, method in enumerate(methods):
            normal_score = df[(df['num_concepts'] == num_concepts) & 
                            (df['method'] == method) & 
                            (df['model_type'] == 'normal')]['spurious_score'].values[0]
            
            x_pos = i * (1 + gap) + j * bar_width
            bar = plt.bar(x_pos, normal_score, bar_width, 
                   label=f'{method} (normal)' if i == 0 else "", 
                   color=normal_color, alpha=0.7,
                   edgecolor=normal_edge, linewidth=2)
            
            # Add rotated text inside bar
            if normal_score > 0.1:  # Only add text if bar is tall enough
                plt.text(x_pos, normal_score/2, f'{method}\n(normal)', 
                        ha='center', va='center', rotation=90,
                        color='black', fontsize=8)
        
        # Then create all bugged bars
        for j, method in enumerate(methods):
            bugged_score = df[(df['num_concepts'] == num_concepts) & 
                            (df['method'] == method) & 
                            (df['model_type'] == 'bugged')]['spurious_score'].values[0]
            
            x_pos = i * (1 + gap) + (j + 3) * bar_width
            bar = plt.bar(x_pos, bugged_score, bar_width, 
                   label=f'{method} (bugged)' if i == 0 else "", 
                   color=bugged_color, alpha=0.7,
                   edgecolor=bugged_edge, linewidth=2)
            
            # Add rotated text inside bar
            if bugged_score > 0.1:  # Only add text if bar is tall enough
                plt.text(x_pos, bugged_score/2, f'{method}\n(bugged)', 
                        ha='center', va='center', rotation=90,
                        color='black', fontsize=8)
    
    # Customize the plot
    plt.xlabel('Number of Concepts')
    plt.ylabel('Spurious Score')
    plt.title('Spurious Scores by Method and Model Type', fontsize=16, fontweight='bold')
    
    # Set x-axis ticks
    plt.xticks([i * (1 + gap) + 2.5 * bar_width for i in range(len(concept_nums))], 
               [f'{n} concepts' for n in concept_nums])
    
    # Move legend back outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_filename = csv_path.rsplit('.', 1)[0] + '_methods_plot.png'
    output_dir = 'final_results/o2o_easy/o2o_easy_graphs/spurious_score_plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(plot_filename))
    print(f"Saving plot to: {output_path}")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py path/to/spurious_scores.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    create_spurious_score_plot(csv_path)