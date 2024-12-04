# Evolution of Spurious Correlation Detection Approaches

## 1. Initial Approach (Early Implementation)
- Used direct model predictions and concept correlations
- Calculated everything on the fly
- Main limitation: Computationally expensive and redundant calculations

## 2. Two-Method Combination (Middle Stage)
- Combined two detection methods:
  1. Importance ratio: comparing concept importance in target vs background
  2. Background correlation: measuring concept correlations with background
- Implemented in early versions of `spurious_detector.py` (now `spurious_detector_legacy.py`)
- Main limitation: Required some manual work that we couldn't do on the model (maintaining separate dataloaders and separation of target versus background)
- Another limitation: Still required real-time computation of concept vectors

## 3. Pre-computed Concept Integration (Current Approach)
### Key Improvements:
1. Uses pre-computed concept vectors and importances from CRAFT/ACE
2. Files structure:
   - `Concepts.npy`: Pre-extracted concept vectors
   - `Importances.npy`: Pre-calculated importance scores
   - `Activations.npy`: Stored activation values
   - `data_list.pt`: Dataset with transforms
   - `data_list_no_transforms.pt`: Original dataset

### Current Workflow:
1. Run concept extraction (CRAFT/ACE) first
2. Save results to files
3. SpuriousDetector loads pre-computed results
4. Calculate spurious scores using:
   ```python
   score = concept_importance * abs(correlation)
   ```

### Advantages of Current Approach:
1. More efficient - no redundant calculations
2. More maintainable - clear separation of concept extraction and analysis
3. Reusable - can analyze multiple models using same concept data
4. Faster execution - loads pre-computed results instead of calculating

## Future Potential Improvements:
1. Add visualization tools for concept analysis
2. Implement more sophisticated correlation metrics
3. Add support for different concept extraction methods
4. Create automated threshold determination