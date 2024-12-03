import numpy as np
from scipy.stats import pearsonr
import torch
import copy
import os

class SpuriousDetector:
    """
    Detects spurious correlations using pre-computed concept vectors and importances.
    """
    def __init__(self, model, results_dir, concept_instance_name, validation_data):
        """
        Initialize detector with paths to pre-computed results
        
        Args:
            model: The model to analyze
            results_dir: Directory containing concept extraction results
            concept_instance_name: Name of the concept extraction instance
            validation_data: Validation dataset for evaluating model performance
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validation_data = validation_data
        
        # Load pre-computed results
        concept_results_dir = os.path.join(results_dir, concept_instance_name)
        self.concepts = np.load(os.path.join(concept_results_dir, "Concepts.npy"))
        self.importances = np.load(os.path.join(concept_results_dir, "Importances.npy"))
        self.activations = np.load(os.path.join(concept_results_dir, "Activations.npy"))
        
        # Load data lists
        self.data_list = torch.load(os.path.join(concept_results_dir, "data_list.pt"))
        self.data_list_no_transforms = torch.load(os.path.join(concept_results_dir, "data_list_no_transforms.pt"))

    def calculate_spurious_score(self, concept_idx):
        """
        Calculate spurious score using pre-computed activations and importances
        """
        # Use pre-computed activations instead of recalculating
        concept_activations = self.activations[:, concept_idx]
        concept_importance = self.importances[concept_idx]
        
        # Calculate correlation with model predictions
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for data in self.data_list:
                img = data[0].unsqueeze(0).to(self.device)
                output = self.model(img)
                pred = torch.softmax(output, dim=1).max().item()
                predictions.append(pred)
        
        correlation = np.corrcoef(concept_activations, predictions)[0, 1]
        
        return {
            'concept_idx': concept_idx,
            'importance': concept_importance,
            'correlation': abs(correlation),
            'combined_score': concept_importance * abs(correlation)
        }

    def identify_spurious_concepts(self, threshold=0.5):
        """
        Identify spurious concepts using pre-computed data
        
        Args:
            threshold: Correlation threshold for identifying spurious concepts
        Returns:
            List of spurious concept indices and their scores
        """
        spurious_concepts = []
        
        for idx in range(len(self.concepts)):
            score = self.calculate_spurious_score(idx)
            if score['combined_score'] > threshold:
                spurious_concepts.append(score)
                
        return sorted(spurious_concepts, key=lambda x: x['combined_score'], reverse=True)

    def save_results(self, results_dir, filename):
        """
        Save spurious correlation analysis results
        """
        scores = [self.calculate_spurious_score(idx) for idx in range(len(self.concepts))]
        np.save(os.path.join(results_dir, f"{filename}_spurious_scores.npy"), scores)