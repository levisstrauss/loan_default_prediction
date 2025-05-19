# Loan Default Prediction MLflow Pipeline

A complete MLflow pipeline for predicting loan defaults using the LendingClub dataset,
featuring XGBoost modeling, feature engineering, model monitoring, and integration with Weights & Biases.

## Project Overview

This project implements an end-to-end machine learning pipeline for loan default prediction.
It's designed as a series of modular MLflow steps, with artifacts tracked and shared via Weights & Biases.
The pipeline follows best practices for production ML systems, including automated feature engineering,
hyperparameter optimization, model calibration, interpretability, and drift monitoring.

## Pipeline Components

1. **Data Download**: Downloads and prepares the LendingClub loan dataset
2. **Basic Cleaning**: Performs initial data cleaning and preprocessing
3. **Feature Engineering**: Applies advanced transformations and generates new predictive features
4. **Data Split**: Divides data into train, validation, and test sets
5. **Model Training**: Trains an XGBoost model with hyperparameter optimization and calibration
6. **Model Evaluation**: Evaluates the model on test data and tracks performance metrics
7. **Model Monitoring**: Sets up a monitoring system to detect data and concept drift

## Requirements

- Python 3.10+
- MLflow 2.8.1+
- Weights & Biases account
- Conda (for environment management)
- Other dependencies as specified in conda.yml files

## Setup

1. Clone this repository:
```bash
   git clone https://github.com/yourusername/loan-default-prediction
   cd loan-default-prediction
```
2. Log in to Weights & Biases:

```bash
   wandb login
```
3. Run the entire pipeline:

```bash
   mlflow run .
```

## Running Individual Steps
To run specific steps of the pipeline:

```bash
   # Run only the download and cleaning steps
    mlflow run . -P steps=download,basic_cleaning

    # Run only the model training step
    mlflow run . -P steps=train_model

    # Run only the evaluation step
    mlflow run . -P steps=evaluate_model
```

## Configuration

The pipeline is configured via the config.yaml file. You can override

any configuration parameter using Hydra's syntax:

```bash
   # Change XGBoost parameters
   mlflow run . -P hydra_options="modeling.xgboost.n_estimators=200 modeling.xgboost.max_depth=8"

    # Change data preprocessing parameters
    mlflow run . -P hydra_options="etl.min_loan_amount=2000 etl.max_loan_amount=35000"

    # Change feature engineering settings
    mlflow run . -P hydra_options="feature_engineering.enable_auto_selection=false"
```

## Project Structure

```bash
    loan_default_prediction/
    │
    ├── MLproject                # Main MLflow project definition
    ├── conda.yml                # Main conda environment
    ├── config.yaml              # Hydra configuration
    ├── main.py                  # Pipeline orchestration
    │
    ├── src/                     # Source code for pipeline steps
    │   ├── download/            # Data download step
    │   ├── eda/                 # Some EDA on the dataset before implementation on the pipeline
    │   ├── basic_cleaning/      # Data cleaning step
    │   ├── feature_engineering/ # Feature engineering step
    │   ├── data_split/          # Data splitting step
    │   ├── train_model/         # Model training step
    │   ├── evaluate_model/      # Model evaluation step
    │   └── model_monitoring/    # Model monitoring step
    │
    └── README.md                # Project documentation
```

## Pipeline Details

## Download Step

Downloads the LendingClub loan data from a remote source and uploads it to W&B as an artifact.

## Basic Cleaning
Performs initial data preprocessing:

- Converts loan status to binary target (0 = paid, 1 = default)
- Removes outliers based on loan amount
- Converts string values to appropriate numeric types
- Handles date columns and creates date-based features

## Feature Engineering
Implements advanced feature engineering:

- Creates financial ratio features (loan-to-income, debt-to-credit, etc.)
- Generates interaction terms between key features
- Applies appropriate scaling and encoding
- Performs automated feature selection

## Data Split
Splits the data into training, validation, and test sets with proper stratification.

## Model Training
Trains an XGBoost model with:

- Hyperparameter optimization using Optuna
- Probability calibration for accurate risk estimates
- Feature importance analysis
- SHAP-based interpretability

## Model Evaluation
Evaluates the model on test data:

- Comprehensive performance metrics
- Visualization of model performance
- Registration of production-ready models

## Model Monitoring
Sets up a monitoring system to detect:

- Data drift in feature distributions
- Concept drift in model behavior
- Changes in feature importance
- Performance degradation

## Dashboard

The model monitoring step generates an interactive dashboard for visualizing model health and drift metrics.
After running the monitoring step, you can find the dashboard at:

```bash
   monitoring/dashboard.html
```
## Contributing
Contributions to this project are welcome! Please feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
