import os
import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Directories - change based on where your project is saved
BASE_DIR = "/content/drive/MyDrive/CS2822/CS2822_Final_Project"
PROJECT_DIR = os.path.join(BASE_DIR, "Notebooks")  # This is where the current file is
MODELS_DIR = os.path.join(BASE_DIR, "Models")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

# Model configurations
MODEL_NAME = "resnet18.a1_in1k"
normal_model_name = "o2o_medium_resnet18.a1-e=2-lr=0.01_limit=20.pt"
bugged_model_name = "o2o_easy_resnet18.a1-e=2-lr=0.01_limit=20.pt"
concept_instance_name = "CRAFT_12_2_24_Corgi_Snow_320_im"

def set_model_name(name):
    global MODEL_NAME
    MODEL_NAME = name

def load_model(model_name, model_type="resnet18.a1_in1k"):
    """Load a pre-trained model from the models directory"""
    checkpoint_path = os.path.join(MODELS_DIR, model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weights = torch.load(checkpoint_path, map_location=device)

    model = timm.create_model(
        model_type,
        pretrained=True,
        num_classes=4
    ).eval()

    model.load_state_dict(model_weights)
    model.eval()
    return model

def load_concept_results(concept_instance_name):
    """Load pre-computed concept results from the results directory"""
    concept_results_dir = os.path.join(RESULTS_DIR, concept_instance_name)
    
    results = {
        'concepts': np.load(os.path.join(concept_results_dir, "Concepts.npy")),
        'importances': np.load(os.path.join(concept_results_dir, "Importances.npy")),
        'activations': np.load(os.path.join(concept_results_dir, "Activations.npy")),
        'data_list': torch.load(os.path.join(concept_results_dir, "data_list.pt")),
        'data_list_no_transforms': torch.load(os.path.join(concept_results_dir, "data_list_no_transforms.pt"))
    }
    
    return results

if __name__ == "__main__":
    # Set up model
    set_model_name(MODEL_NAME)
    
    # Load models
    try:
        normal_model = load_model(normal_model_name)
        bugged_model = load_model(bugged_model_name)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        
    # Load pre-computed results
    try:
        results = load_concept_results(concept_instance_name)
        print("Pre-computed results loaded successfully")
        print(f"Number of concepts: {len(results['concepts'])}")
        print(f"Number of images: {len(results['data_list'])}")
        
        # Display a sample image
        if len(results['data_list_no_transforms']) > 0:
            plt.figure(figsize=(8, 8))
            plt.imshow(results['data_list_no_transforms'][0][0])
            plt.title("Sample Image")
            plt.axis('off')
            plt.show()
            
    except Exception as e:
        print(f"Error loading results: {e}")