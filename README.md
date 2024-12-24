# AgriField-Performance-Classifier-Challenge
This project implements a machine learning pipeline to classify agricultural field performance into categories (`low`, `medium`, `high`) based on various farm-related features. The solution is built using Python and employs a `RandomForestClassifier` from `scikit-learn` to perform the classification task. The implementation also includes data preprocessing, feature engineering, and hyperparameter tuning.

## Dataset

The dataset consists of:
- **Training data** (`train.csv`): Contains labeled examples for supervised learning.
- **Testing data** (`test.csv`): Used for generating predictions.

### Key Features
1. **Features**: Various characteristics of agricultural fields such as farm equipment area, field size, irrigation system type, etc.
2. **Target**: The performance category of the field (`low`, `medium`, `high`).

## Project Workflow

### 1. Libraries Used
The following libraries are utilized:
- `pandas`: For data manipulation.
- `numpy`: For numerical operations.
- `scikit-learn`: For machine learning pipeline, model training, evaluation, and hyperparameter tuning.

### 2. Data Preprocessing
- **Feature Removal**: Columns with more than 85,000 null values are removed from both training and testing datasets.
- **Missing Value Imputation**: Missing values in the remaining features are replaced with the median of the respective columns.
- **Target Encoding**: The target variable (`Target`) is encoded as follows:
  - `low`: 0
  - `medium`: 1
  - `high`: 2
- **Feature Scaling**: `StandardScaler` is used to normalize features.

### 3. Model Pipeline
A pipeline is created using `scikit-learn` which includes:
1. **Scaler**: Standardizes the dataset using `StandardScaler`.
2. **Classifier**: `RandomForestClassifier` is used as the model.

### 4. Hyperparameter Tuning
- **Grid Search**: Parameters for the `RandomForestClassifier` are tuned using `GridSearchCV`.
- **Evaluation Metric**: `f1_score` with `macro` averaging is used for model evaluation.

### 5. Model Evaluation
- The dataset is split into training and validation sets using `train_test_split` with stratified sampling to handle class imbalance.
- Cross-validation is performed with `StratifiedKFold`.
- The model is evaluated on the validation set using the F1 macro score.

### 6. Prediction and Submission
- Predictions are made on the test dataset after model training.
- Predicted labels are converted back to their original categories (`low`, `medium`, `high`).
- Results are saved in a CSV file (`RandomForest.csv`) for submission.

## Steps to Run the Code

1. **Prerequisites**:
   - Install Python 3.x.
   - Install the required libraries using:
     ```bash
     pip install pandas numpy scikit-learn
     ```

2. **Run the Script**:
   - Place the `train.csv` and `test.csv` files in the same directory as the script.
   - Execute the script:
     ```bash
     python AgriField-Performance-Classifier-Challenge.py
     ```

3. **Output**:
   - The script generates a file named `RandomForest.csv` containing the predictions.

## File Descriptions

- `AgriField-Performance-Classifier-Challenge.py`: Main script implementing the solution.
- `train.csv`: Training dataset.
- `test.csv`: Testing dataset.
- `RandomForest.csv`: Output file containing predictions for the test set.

## Key Parameters and Best Model

- **Hyperparameters**:
  - `n_estimators`: 224
  - `max_depth`: 22
  - `min_samples_split`: 6
  - `min_samples_leaf`: 10
  - `class_weight`: `balanced_subsample`

- **Best Model**: Random forest classifier with the above parameters.

## Results
The final model achieved an F1 macro score of `0.436` on the validation set. The predictions for the test dataset have been stored in `RandomForest.csv`.

## Improvements
Possible areas for improvement include:
- Feature engineering to derive additional useful features.
- Experimenting with different models and ensemble methods.
- Addressing class imbalance using techniques like oversampling or SMOTE.
- Analyzing feature importance to reduce dimensionality.

## Acknowledgments
This project was developed as part of the AgriField Performance Classifier Challenge. Thanks to the organizers for providing the datasets and problem statement.

---
**Author**: Vinaykumar Kadari 
**Date**: 24 Dec 2024
