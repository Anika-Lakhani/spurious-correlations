import os
import torch
import torch.optim as optim
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import tarfile
from typing import Any, Tuple

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Local paths instead of Google Drive
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
DATASETS_DIR = os.path.join(PROJECT_DIR, "Datasets")
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")

# Create directories if they don't exist
for directory in [MODELS_DIR, DATASETS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# CHANGE THESE based on your model files
MODEL_NAME = "resnet18.a1_in1k"
normal_model_name = "o2o_medium_resnet18.a1-e=2-lr=0.01_limit=20.pt"
bugged_model_name = "o2o_easy_resnet18.a1-e=2-lr=0.01_limit=20.pt"
concept_instance_name = "CRAFT_12_2_24_Corgi_Snow_320_im"

def set_model_name(name):
    global MODEL_NAME
    MODEL_NAME = name

def load_model(model_name, model_type="resnet18.a1_in1k"):
    checkpoint_path = os.path.join(MODELS_DIR, model_name)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found at {checkpoint_path}")

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

def extract_dataset(dataset_name):
    tar_path = os.path.join(DATASETS_DIR, f"{dataset_name}.tar.gz")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Dataset archive not found at {tar_path}")
    
    extract_path = os.path.join(DATASETS_DIR, "spawrious224/1/snow/")
    os.makedirs(extract_path, exist_ok=True)
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

class CustomImageFolder(Dataset):
    def __init__(self, folder_path, class_index, location_index, limit=None, transform=None):
        self.folder_path = folder_path
        self.class_index = class_index
        self.location_index = location_index
        self.image_paths = [
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        class_label = torch.tensor(self.class_index, dtype=torch.long)
        location_label = torch.tensor(self.location_index, dtype=torch.long)
        return img, class_label, location_label

class MultipleDomainDataset:
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 8
    ENVIRONMENTS = None
    INPUT_SHAPE = None

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

def build_combination(benchmark_type, group, test, filler=None):
    total = 3168
    combinations = {}
    if "m2m" in benchmark_type:
        counts = [total, total]
        combinations["train_combinations"] = {
            ("bulldog",): [(group[0], counts[0]), (group[1], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[0], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[3], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[2], counts[1])],
        }
        combinations["test_combinations"] = {
            ("bulldog",): [test[0], test[1]],
            ("dachshund",): [test[1], test[0]],
            ("labrador",): [test[2], test[3]],
            ("corgi",): [test[3], test[2]],
        }
    else:
        counts = [int(0.97 * total), int(0.87 * total)]
        combinations["train_combinations"] = {
            ("bulldog",): [(group[0], counts[0]), (group[0], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[1], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[2], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[3], counts[1])],
            ("bulldog", "dachshund", "labrador", "corgi"): [
                (filler, total - counts[0]),
                (filler, total - counts[1]),
            ],
        }
        combinations["test_combinations"] = {
            ("bulldog",): [test[0], test[0]],
            ("dachshund",): [test[1], test[1]],
            ("labrador",): [test[2], test[2]],
            ("corgi",): [test[3], test[3]],
        }
    return combinations

def _get_combinations(benchmark_type: str) -> Tuple[dict, dict]:
    combinations = {
        "o2o_easy": (
            ["desert", "jungle", "dirt", "snow"],
            ["dirt", "snow", "desert", "jungle"],
            "beach",
        ),
        "o2o_medium": (
            ["mountain", "beach", "dirt", "jungle"],
            ["jungle", "dirt", "beach", "snow"],
            "desert",
        ),
        "o2o_hard": (
            ["jungle", "mountain", "snow", "desert"],
            ["mountain", "snow", "desert", "jungle"],
            "beach",
        ),
        "m2m_hard": (
            ["dirt", "jungle", "snow", "beach"],
            ["snow", "beach", "dirt", "jungle"],
            None,
        ),
        "m2m_easy": (
            ["desert", "mountain", "dirt", "jungle"],
            ["dirt", "jungle", "mountain", "desert"],
            None,
        ),
        "m2m_medium": (
            ["beach", "snow", "mountain", "desert"],
            ["desert", "mountain", "beach", "snow"],
            None,
        ),
    }
    if benchmark_type not in combinations:
        raise ValueError("Invalid benchmark type")
    group, test, filler = combinations[benchmark_type]
    return build_combination(benchmark_type, group, test, filler)

if __name__ == "__main__":
    # Set up model
    set_model_name(MODEL_NAME)
    
    # Extract dataset if needed
    dataset_name = "1-snow-corgi"
    extract_dataset(dataset_name)
    
    # Load models
    try:
        normal_model = load_model(normal_model_name)
        bugged_model = load_model(bugged_model_name)
        print("Models loaded successfully")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        exit(1)
    
    # Locate results directory for concept extraction
    concept_results_dir = os.path.join(RESULTS_DIR, concept_instance_name)
    os.makedirs(concept_results_dir, exist_ok=True)
    
    # Load pre-computed results if they exist
    try:
        concepts = np.load(os.path.join(concept_results_dir, "Concepts.npy"))
        importances = np.load(os.path.join(concept_results_dir, "Importances.npy"))
        data_list = torch.load(os.path.join(concept_results_dir, "data_list.pt"))
        data_list_no_transforms = torch.load(os.path.join(concept_results_dir, "data_list_no_transforms.pt"))
        activations = np.load(os.path.join(concept_results_dir, "Activations.npy"))
        print("Pre-computed results loaded successfully")
    except FileNotFoundError:
        print("Pre-computed results not found. Please run concept extraction first.")
        exit(1)

    # Example: Check that images were loaded correctly
    if len(data_list_no_transforms) > 15:
        plt.imshow(data_list_no_transforms[15][0])
        plt.show()
