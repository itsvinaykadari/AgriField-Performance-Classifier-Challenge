"""
================================================================================
                    FARM PERFORMANCE CLASSIFICATION SYSTEM
================================================================================

PURPOSE:
    This module implements a machine learning pipeline to classify agricultural
    field performance into three categories: low, medium, and high. It processes
    agricultural field data, handles missing values, and employs a Random Forest
    Classifier with optimized hyperparameters for accurate predictions.

PROBLEM STATEMENT:
    Agricultural productivity assessment is crucial for farm management and
    resource optimization. Manual evaluation of farm performance is time-consuming
    and subjective. This solution automates the classification of farm performance
    based on quantitative features such as equipment area, field size, irrigation
    systems, soil fertility, and other agricultural parameters. By leveraging
    machine learning, stakeholders can quickly identify underperforming farms and
    take corrective actions to improve overall productivity.

WORKFLOW:
    1. Data Loading: Read training and testing datasets from CSV files
    2. Data Cleaning: Remove features with excessive missing values (>85%)
    3. Missing Value Imputation: Fill remaining nulls with median values
    4. Feature Engineering: Separate features and target variable
    5. Data Splitting: Split into training and validation sets (80-20 split)
    6. Model Training: Train Random Forest with hyperparameter tuning
    7. Evaluation: Assess model performance using F1 macro score
    8. Prediction: Generate predictions on test dataset
    9. Output: Save results to CSV file for submission

OUTPUT:
    CSV file containing UID and predicted performance categories for each farm

================================================================================
"""

# Importing the libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main(test_fname):
    # Loading dataset in the train and test by using pandas
    training_dt = pd.read_csv("train.csv")
    testing_dt = pd.read_csv(test_fname)
    training_dt.head()
    training_dt.shape # get the shape of the training sets  testing_dt.shape
    testing_dt.shape # get the shape of the testing sets
    training_dt.isnull().sum() # give total number of null values present in each column  
    
    # dropping the column which contain more than 85000 null values in training sets
    training_dt.drop(['CropFieldConfiguration','CultivatedAndWildArea','FarmClassification','FarmShedAreaSqft','FieldConstructionType','FieldShadeCover','FieldZoneLevel',
                      'HarvestStorageSqft','HasGreenHouse','HasPestControl','NaturalLakePresence','NumberGreenHouses','PartialIrrigationSystemCount','PerimeterGuardPlantsArea','PrimaryCropAreaSqft',
                      'PrimaryCropAreaSqft2','ReservoirType','ReservoirWithFilter','TaxOverdueStatus','TaxOverdueYear','TotalAreaSqft','TotalReservoirSize','UndergroundStorageSqft'],axis=1,inplace=True)

    training_dt.shape # get the shape of the training sets  testing_dt.isnull().sum()/testing_dt.shape[0]*100
    testing_dt.isnull().sum()/testing_dt.shape[0]*100 # give total number of null values present in each column in the form of percentage
    # dropping the column which contain more than 85000 null values in testing sets

    testing_dt.drop(['CropFieldConfiguration','CultivatedAndWildArea','FarmClassification','FarmShedAreaSqft','FieldConstructionType','FieldShadeCover','FieldZoneLevel',
                     'HarvestStorageSqft','HasGreenHouse','HasPestControl','NaturalLakePresence','NumberGreenHouses','PartialIrrigationSystemCount','PerimeterGuardPlantsArea','PrimaryCropAreaSqft',
                     'PrimaryCropAreaSqft2','ReservoirType','ReservoirWithFilter','TaxOverdueStatus','TaxOverdueYear','TotalAreaSqft','TotalReservoirSize','UndergroundStorageSqft'],axis=1,inplace=True)

    training_dt.info() # get me the information of the each column like what dtype, count non-null etc  # Impute missing values using median

    # Handle Missing Value by using median imputation
    training_dt['FarmEquipmentArea'] = training_dt['FarmEquipmentArea'].fillna(training_dt['FarmEquipmentArea'].median())
    training_dt['FarmVehicleCount'] = training_dt['FarmVehicleCount'].fillna(training_dt['FarmVehicleCount'].median())
    training_dt['FarmingCommunityId'] = training_dt['FarmingCommunityId'].fillna(training_dt['FarmingCommunityId'].median())
    training_dt['FarmingUnitCount'] = training_dt['FarmingUnitCount'].fillna(training_dt['FarmingUnitCount'].median())
    training_dt['FieldSizeSqft'] = training_dt['FieldSizeSqft'].fillna(training_dt['FieldSizeSqft'].median())
    training_dt['HarvestProcessingType'] = training_dt['HarvestProcessingType'].fillna(training_dt['HarvestProcessingType'].median())
    training_dt['NumberOfFarmingZones'] = training_dt['NumberOfFarmingZones'].fillna(training_dt['NumberOfFarmingZones'].median())
    training_dt['OtherZoningCode'] = training_dt['OtherZoningCode'].fillna(training_dt['OtherZoningCode'].median())
    training_dt['SoilFertilityType'] = training_dt['SoilFertilityType'].fillna(training_dt['SoilFertilityType'].median())
    training_dt['TypeOfIrrigationSystem'] = training_dt['TypeOfIrrigationSystem'].fillna(training_dt['TypeOfIrrigationSystem'].median())
    training_dt['WaterReservoirCount'] = training_dt['WaterReservoirCount'].fillna(training_dt['WaterReservoirCount'].median())

    testing_dt.isnull().sum() # give total number of null values present in each column  # Impute missing values in testing_dt using median
    
    # Handle Missing Value by using median imputation
    testing_dt['FarmEquipmentArea'] = testing_dt['FarmEquipmentArea'].fillna(testing_dt['FarmEquipmentArea'].median())
    testing_dt['FarmVehicleCount'] = testing_dt['FarmVehicleCount'].fillna(testing_dt['FarmVehicleCount'].median())
    testing_dt['FarmingCommunityId'] = testing_dt['FarmingCommunityId'].fillna(testing_dt['FarmingCommunityId'].median())
    testing_dt['FarmingUnitCount'] = testing_dt['FarmingUnitCount'].fillna(testing_dt['FarmingUnitCount'].median())
    testing_dt['FieldSizeSqft'] = testing_dt['FieldSizeSqft'].fillna(testing_dt['FieldSizeSqft'].median())
    testing_dt['HarvestProcessingType'] = testing_dt['HarvestProcessingType'].fillna(testing_dt['HarvestProcessingType'].median())
    testing_dt['NumberOfFarmingZones'] = testing_dt['NumberOfFarmingZones'].fillna(testing_dt['NumberOfFarmingZones'].median())
    testing_dt['OtherZoningCode'] = testing_dt['OtherZoningCode'].fillna(testing_dt['OtherZoningCode'].median())
    testing_dt['SoilFertilityType'] = testing_dt['SoilFertilityType'].fillna(testing_dt['SoilFertilityType'].median())
    testing_dt['TypeOfIrrigationSystem'] = testing_dt['TypeOfIrrigationSystem'].fillna(testing_dt['TypeOfIrrigationSystem'].median())
    testing_dt['WaterReservoirCount'] = testing_dt['WaterReservoirCount'].fillna(testing_dt['WaterReservoirCount'].median())

    testing_dt.isnull().sum() # give total number of null values present in each column  # Separate features and target
    
    # Separating training set and setting features to X and Y to target
    X = training_dt.drop(['UID', 'Target'], axis=1)  # dropping UID and Target
    y = training_dt['Target']
    
    # Converting object dtype to int
    y = y.map({'low': 0, 'medium': 1, 'high': 2})

    # F1 macro scorer
    f1_macro_scoreer = make_scorer(f1_score, average='macro')  # Split data to avoid overfitting on cross-validation and separate validation set
    
    # Split data to avoid overfitting on cross-validation and separate validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Defining the (param_grid) criteria the model should trained
    param_grid = {
        'randomforestclassifier__n_estimators': [224],
        'randomforestclassifier__max_depth': [22],
        'randomforestclassifier__min_samples_split': [6],
        'randomforestclassifier__min_samples_leaf': [10],
        'randomforestclassifier__class_weight': ['balanced_subsample']
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('randomforestclassifier', RandomForestClassifier(random_state=42))
    ])

    # for imbalanced data stratified the k fold classification
    sf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Using grid search to get the parameter
    grid_search_cv = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=f1_macro_scoreer, cv=sf, n_jobs=-1, verbose=2)
    grid_search_cv.fit(X_train, y_train)

    # after grid search we get best parameter which will be our best model
    bestmodel = grid_search_cv.best_estimator_

    # printing
    print("Best parameters found:", grid_search_cv.best_params_)
    
    # Evaluting the model on our validation sets
    y_value_pred = bestmodel.predict(X_val)
    value_f1_sc = f1_score(y_val, y_value_pred, average='macro')
    print("F1 score:", value_f1_sc)

    # Predicting my model based on test sets
    X_test = testing_dt.drop(['UID'], axis=1)
    test_predd = bestmodel.predict(X_test)

    # Convert predictions back to labels
    test_predictions_labels = pd.Series(test_predd).map({0: 'low', 1: 'medium', 2: 'high'})

    # Storing the result in the submission.csv file
    submission = pd.DataFrame({'UID': testing_dt['UID'], 'Target': test_predictions_labels})
    # submission.to_csv('RandomForest.csv', index=False)
    return submission

import argparse
def make_predictions(test_fname, predictions_fname):
#TODO: complete this function to save predictions to the csv file predictions_fname
#this is an example, you need to modify the code below to fit your workflow
# #### start code ####
#   test = pd.read_csv(test_fname)
#   fill_na_values(test, features, vals)
#   test_X = test[features].to_numpy()
#   preds = model.predict(test_X)
#   test_uid = test[["UID"]].copy()
#   test_uid["Target"] = preds.reshape(-1)
#   test_uid.to_csv(predictions_fname, index=False)
    output = main(test_fname)
    output.to_csv(predictions_fname, index=False)
#### end code ####
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, help='file path of train.csv')
    parser.add_argument("--test-file", type=str, help='file path of test.csv')
    parser.add_argument("--predictions-file", type=str, help='save path of predictions')
    args = parser.parse_args()
    make_predictions(args.test_file, args.predictions_file)

