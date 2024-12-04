import numpy as np
from scipy.stats import pearsonr
import torch
import copy

"""
1. dog_data: DataLoader containing the target class images (corgi/dog images)
- This would be the primary class we're interested in
- Used to calculate the "true" importance of concepts

2. background_data: DataLoader containing images from other environments/contexts
- Images where we shouldn't see the concept if it's not spurious
- For example, if analyzing "snow" as a concept for corgis:
    - dog_data would be corgi images
    - background_data would be non-snow environment images (desert, jungle, etc.)
"""

class SpuriousDetector:
    """
    Detects and removes spurious correlations in a neural network model by analyzing concept importance
    and background correlations.
    """
    def __init__(self, model, concept_vectors, concept_importances, validation_data):
        self.model = model
        self.concepts = concept_vectors
        self.importances = concept_importances
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validation_data = validation_data

    ###########################################
    # Main Public Methods
    ###########################################
    
    def calculate_spurious_score(self, concept_idx, dog_data, background_data):
        """
        Main method to calculate combined spurious score using both approaches.
       
       Basically, we are combining two methods to calculate how spurious a concept is:
        1. Importance ratio: comparing concept importance in target vs background
        2. Background correlation: how much the concept correlates with background
        If you want to just use one method, you can use the helper methods directly
        (located in the Spurious Score Calculation Helpers section).
        """
        importance_ratio = self.calculate_importance_ratio(concept_idx, dog_data, background_data)
        bg_correlation = self.calculate_background_correlation(concept_idx, background_data)
        
        return {
            'concept_idx': concept_idx,
            'importance_ratio': importance_ratio,
            'bg_correlation': bg_correlation,
            'combined_score': importance_ratio * bg_correlation
        }
    
    def remove_spurious_concepts(self, threshold):
        """Main method to iteratively remove concepts until desired accuracy threshold is met"""
        scores = []
        for idx in range(len(self.concepts)):
            score = self.calculate_spurious_score(idx, self.dog_data, self.bg_data)
            scores.append(score)
            
        sorted_concepts = sorted(scores, key=lambda x: x['combined_score'], reverse=True)
        return self.filter_concepts(sorted_concepts, threshold)

    ###########################################
    # Spurious Score Calculation Helpers
    ###########################################
    
    def calculate_importance_ratio(self, concept_idx, dog_data, background_data):
        """
        Calculate the importance ratio for a concept.
        Can be used alone if you just want to use this method for spurious score instead of combining
        with the below background correlation method.
        """
        dog_importance = self.get_concept_importance(concept_idx, dog_data)
        bg_importance = self.get_concept_importance(concept_idx, background_data)
        return bg_importance / dog_importance
    
    def calculate_background_correlation(self, concept_idx, background_data):
        """
        Calculate correlation between concept activations and predictions on background data.
        Can be used alone if you just want to use this method for spurious score instead of combining
        with the above importance ratio method.
        """
        concept_activations = []
        prediction_confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in background_data:
                images = batch[0].to(self.device)
                activations = self.get_concept_activations(images, concept_idx)
                concept_activations.extend(activations.cpu().numpy())
                
                outputs = self.model(images)
                confidences = torch.softmax(outputs, dim=1)
                max_confidences = confidences.max(dim=1).values
                prediction_confidences.extend(max_confidences.cpu().numpy())
        
        correlation = np.corrcoef(concept_activations, prediction_confidences)[0, 1]
        return abs(correlation)

    ###########################################
    # Concept Importance Calculation Helpers (MIGHT NOT NEED THIS WHOLE SECTION DEPENDING ON WHAT THOMAS GIVES ME)
    ###########################################
    
    def get_concept_importance(self, concept_idx, data):
        """Helper: Calculate importance of a concept for a given dataset"""
        importance_scores = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in data:
                images = batch[0].to(self.device)
                concept_activations = self.get_concept_activations(images, concept_idx)
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                batch_importance = self.calculate_batch_importance(
                    concept_activations,
                    outputs,
                    predictions
                )
                importance_scores.append(batch_importance)
        
        return torch.mean(torch.tensor(importance_scores)).item()
    
    def get_concept_activations(self, images, concept_idx):
        """Helper: Calculate activation values for a specific concept across all images"""
        concept = self.concepts[concept_idx]
        features = self.get_intermediate_features(images)
        activations = torch.matmul(features, concept)
        return activations
    
    def calculate_batch_importance(self, concept_activations, outputs, predictions):
        """Helper: Calculate importance score for a single batch"""
        confidences = torch.softmax(outputs, dim=1)
        pred_confidences = confidences[torch.arange(len(predictions)), predictions]
        
        correlation = torch.corrcoef(
            torch.stack([concept_activations, pred_confidences])
        )[0,1]
        
        return abs(correlation.item())
    
    def get_intermediate_features(self, images):
        """
        Helper: Extract intermediate layer features from the model
        
        idk why we need this function but AI generated it
        and it might be helpful to understand inner layers of the model
        """
        features = None
        
        def hook(module, input, output):
            nonlocal features
            features = output.flatten(start_dim=1)
            
        # Register hook on the appropriate layer (modify for specific model architecture)
        handle = self.model.layer4.register_forward_hook(hook)
        self.model(images)
        handle.remove()
        
        return features

    ###########################################
    # Concept Filtering and Model Modification
    ###########################################
        
    def filter_concepts(self, sorted_concepts, threshold):
        """Helper: Iteratively remove concepts until model reaches accuracy threshold"""
        current_model = self.model
        removed_concepts = []
        current_accuracy = self.evaluate_model(current_model)
        
        if current_accuracy >= threshold:
            return current_model, removed_concepts, current_accuracy
        
        for concept in sorted_concepts:
            concept_idx = concept['concept_idx']
            current_model = self.mask_concept(current_model, concept_idx)
            removed_concepts.append(concept)
            
            new_accuracy = self.evaluate_model(current_model)
            print(f"Removed concept {concept_idx}, new accuracy: {new_accuracy:.3f}")
            
            if new_accuracy >= threshold:
                return current_model, removed_concepts, new_accuracy
                
            if new_accuracy < current_accuracy:
                current_model = self.unmask_concept(current_model, concept_idx)
                removed_concepts.pop()
            else:
                current_accuracy = new_accuracy
        
        return current_model, removed_concepts, current_accuracy
    
    def evaluate_model(self, model):
        """Helper: Evaluate model accuracy on validation dataset"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.validation_data:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def mask_concept(self, model, concept_idx):
        """Helper: Create a copy of the model with the specified concept masked out"""
        masked_model = copy.deepcopy(model)
        
        with torch.no_grad():
            # Zero out the concept's influence in the final layer
            masked_model.final_layer.weight.data[:, concept_idx] = 0
        
        return masked_model
    
    def unmask_concept(self, model, concept_idx):
        """Helper: Restore a previously masked concept"""
        unmasked_model = copy.deepcopy(model)
        
        with torch.no_grad():
            unmasked_model.final_layer.weight.data[:, concept_idx] = \
                self.model.final_layer.weight.data[:, concept_idx].clone()
        
        return unmasked_model