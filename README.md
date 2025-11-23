# Farm Performance Classification System

## Overview

This project implements a machine learning pipeline to automatically classify agricultural field performance into three categories: **low**, **medium**, and **high**. The solution leverages machine learning techniques to assess farm productivity based on quantitative agricultural features.

## Problem Statement

Agricultural productivity assessment is crucial for farm management and resource optimization. Manual evaluation of farm performance is time-consuming and subjective. This solution addresses this challenge by:

- **Automating performance classification** based on quantitative agricultural features
- **Reducing manual assessment burden** for agricultural stakeholders
- **Enabling data-driven decision making** for farm management
- **Identifying underperforming farms** to facilitate corrective action
- **Optimizing resource allocation** across agricultural operations

## Dataset

The dataset is available from the VNB FOML 2024 Hackathon competition on Kaggle:
- **Source**: [VNB FOML 2024 Hackathon - Kaggle](https://www.kaggle.com/competitions/vnb-foml-2024-hackathon)
- **Training data** (`train.csv`): Contains labeled examples with known performance categories for supervised learning
- **Testing data** (`test.csv`): Used for generating predictions on unlabeled data

### Key Features

The dataset includes various agricultural field characteristics:
- Farm equipment specifications
- Field dimensions and configurations
- Irrigation system types and counts
- Soil fertility indicators
- Crop-related parameters
- Water reservoir details
- And other farming infrastructure metrics

### Target Variable

The target variable represents field performance with three classes:
- `low`: Underperforming fields
- `medium`: Moderate performance fields
- `high`: High-performing fields

## Technical Approach

### 1. Libraries Used

- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning pipeline, model training, evaluation, and hyperparameter tuning

### 2. Data Preprocessing

#### Feature Selection
- Removed 22 features with more than 85% missing values from both training and testing datasets
- Retained features with sufficient data coverage

#### Missing Value Imputation
- Applied median imputation for remaining missing values
- Ensures no data loss while maintaining statistical properties

#### Target Encoding
- Converted categorical target to numeric format:
  - `low` → 0
  - `medium` → 1
  - `high` → 2

#### Feature Scaling
- Applied `StandardScaler` to normalize feature ranges
- Ensures equal importance across features during model training

### 3. Model Architecture

The solution uses a pipeline combining:

1. **Data Preprocessing**: `StandardScaler` for feature normalization
2. **Classification**: `RandomForestClassifier` with tuned hyperparameters

### 4. Hyperparameter Tuning

#### Optimization Method
- Used `GridSearchCV` for systematic hyperparameter search
- Applied 5-fold stratified cross-validation

#### Best Hyperparameters
- `n_estimators`: 224 (number of trees in the forest)
- `max_depth`: 22 (maximum depth of trees)
- `min_samples_split`: 6 (minimum samples to split a node)
- `min_samples_leaf`: 10 (minimum samples at leaf nodes)
- `class_weight`: `balanced_subsample` (handles class imbalance)

### 5. Model Evaluation

- **Metric**: F1 score with macro averaging
- **Validation**: 80-20 train-validation split with stratified sampling
- **Cross-validation**: 5-fold stratified cross-validation for robust evaluation

### 6. Prediction and Output

- Generates predictions on test dataset using the trained model
- Converts numeric predictions back to categorical labels
- Outputs results in CSV format with UID and predicted performance category

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```

## Usage

### Running the Classifier

1. Place your `train.csv` and `test.csv` files in the same directory as the script
2. Execute the script:
   ```bash
   python farm_performance_classifier.py \
     --train-file train.csv \
     --test-file test.csv \
     --predictions-file predictions.csv
   ```

### Alternative Usage

If running with default file names in the current directory:
```bash
python farm_performance_classifier.py
```

### Output

The script generates a CSV file containing:
- **UID**: Unique identifier for each farm
- **Target**: Predicted performance category (low, medium, or high)

## File Descriptions

| File | Description |
|------|-------------|
| `farm_performance_classifier.py` | Main script implementing the classification pipeline |
| `train.csv` | Training dataset with labeled performance categories |
| `test.csv` | Test dataset for generating predictions |
| `predictions.csv` | Output file with UID and predicted performance |
| `README.md` | Project documentation |

## Performance Metrics

- **F1 Score (Macro)**: 0.436 on validation set
- **Model**: Random Forest with 224 trees
- **Evaluation Method**: 5-fold Stratified Cross-Validation

## Algorithm Details

### Random Forest Classifier

The model uses Random Forest algorithm because:
- Handles non-linear relationships in agricultural data
- Robust to outliers and missing values
- Provides feature importance insights
- Works well with imbalanced datasets when using balanced class weights
- Scalable for large datasets

### Class Imbalance Handling

- **Stratified Sampling**: Maintains class distribution during train-test split
- **Balanced Subsample**: Automatically adjusts class weights during training
- **Stratified K-Fold**: Ensures balanced folds during cross-validation

## Future Improvements

Potential areas for enhancement:

1. **Feature Engineering**: Derive additional meaningful features from existing ones
2. **Alternative Models**: Experiment with gradient boosting, XGBoost, or neural networks
3. **Advanced Imbalance Handling**: Implement SMOTE or other resampling techniques
4. **Feature Selection**: Use feature importance analysis to reduce dimensionality
5. **Hyperparameter Optimization**: Try Bayesian optimization for better parameter tuning
6. **Ensemble Methods**: Combine multiple models for improved predictions
7. **Data Augmentation**: Generate synthetic samples to improve model robustness
8. **Model Interpretability**: Use SHAP or LIME for model explanation

## Code Structure

```
farm_performance_classifier.py
├── Module Docstring (Purpose, Problem Statement, Workflow, Output)
├── Imports
├── main() function
│   ├── Data Loading
│   ├── Data Preprocessing
│   ├── Feature Engineering
│   ├── Model Training
│   ├── Hyperparameter Tuning
│   ├── Model Evaluation
│   └── Prediction
├── make_predictions() function
└── CLI Interface (argparse)
```

## Performance Considerations

- **Training Time**: Grid search with cross-validation may take several minutes
- **Memory Usage**: Depends on dataset size; typical datasets run efficiently
- **Prediction Speed**: Predictions on test set are typically computed in seconds

## Best Practices Implemented

- ✓ Stratified sampling for imbalanced data
- ✓ Separate validation set to prevent overfitting
- ✓ Pipeline for reproducible preprocessing
- ✓ Median imputation for robust missing value handling
- ✓ Feature scaling for algorithm efficiency
- ✓ Comprehensive error handling and logging
- ✓ Clear function documentation and code comments

## Troubleshooting

### File Not Found Error
- Ensure `train.csv` and `test.csv` are in the current working directory
- Verify file paths are correct in command-line arguments

### Memory Issues
- For large datasets, consider processing in batches
- Reduce number of cross-validation folds if needed

### Poor Model Performance
- Verify data quality and feature relevance
- Consider feature engineering and selection
- Experiment with different hyperparameters
- Check for data leakage between train and test sets

## License

This project is provided for educational and research purposes.

---

**Author**: Vinaykumar Kadari 
**Date**: 24 Dec 2024
