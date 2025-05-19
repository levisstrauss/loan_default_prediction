#!/usr/bin/env python
"""
Feature engineering script for the loan default prediction pipeline.
"""
import argparse
import logging
import pandas as pd
import numpy as np
import wandb
import pickle
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer,
    OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create financial and credit-related features.
    """

    def __init__(self, enable_ratios=True, enable_interactions=True):
        self.enable_ratios = enable_ratios
        self.enable_interactions = enable_interactions
        self.new_feature_names_ = []

    def fit(self, X, y=None):
        self.input_features_ = X.columns.tolist()
        self.numeric_features_ = X.select_dtypes(include=['number']).columns.tolist()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        self.new_feature_names_ = []

        # Create financial ratio features
        if self.enable_ratios:
            # Debt-to-Income related ratios
            if 'loan_amnt' in X.columns and 'annual_inc' in X.columns:
                X_transformed['loan_to_income'] = X['loan_amnt'] / (X['annual_inc'] + 1)
                self.new_feature_names_.append('loan_to_income')

            # Payment to income ratio
            if 'installment' in X.columns and 'annual_inc' in X.columns:
                X_transformed['installment_to_income'] = (X['installment'] * 12) / (X['annual_inc'] + 1)
                self.new_feature_names_.append('installment_to_income')

            # Revolving utilization to income
            if 'revol_bal' in X.columns and 'annual_inc' in X.columns:
                X_transformed['revol_bal_to_income'] = X['revol_bal'] / (X['annual_inc'] + 1)
                self.new_feature_names_.append('revol_bal_to_income')

            # Debt to credit line ratio
            if 'revol_bal' in X.columns and 'total_acc' in X.columns:
                X_transformed['debt_to_credit_lines'] = X['revol_bal'] / (X['total_acc'] + 1)
                self.new_feature_names_.append('debt_to_credit_lines')

            # Credit utilization intensity
            if 'revol_util' in X.columns and 'open_acc' in X.columns:
                X_transformed['util_per_account'] = X['revol_util'] / (X['open_acc'] + 1)
                self.new_feature_names_.append('util_per_account')

            # Payment to loan ratio (affordability measure)
            if 'installment' in X.columns and 'loan_amnt' in X.columns:
                X_transformed['payment_to_loan'] = X['installment'] / (X['loan_amnt'] + 1)
                self.new_feature_names_.append('payment_to_loan')

        # Create feature interactions
        if self.enable_interactions:
            # Interact dti with int_rate
            if 'dti' in X.columns and 'int_rate' in X.columns:
                X_transformed['dti_x_int_rate'] = X['dti'] * X['int_rate']
                self.new_feature_names_.append('dti_x_int_rate')

            # Interact loan_amnt with term
            if 'loan_amnt' in X.columns and 'term' in X.columns:
                X_transformed['loan_amnt_x_term'] = X['loan_amnt'] * X['term']
                self.new_feature_names_.append('loan_amnt_x_term')

            # Interact revol_util with total_acc
            if 'revol_util' in X.columns and 'total_acc' in X.columns:
                X_transformed['revol_util_x_total_acc'] = X['revol_util'] * X['total_acc']
                self.new_feature_names_.append('revol_util_x_total_acc')

        # Create squared terms for selected features
        for feat in ['dti', 'revol_util', 'int_rate']:
            if feat in X.columns:
                X_transformed[f'{feat}_squared'] = X[feat] ** 2
                self.new_feature_names_.append(f'{feat}_squared')

        # Create composite risk score
        risk_columns = ['dti', 'delinq_2yrs', 'revol_util', 'pub_rec']
        available_risk_cols = [col for col in risk_columns if col in X.columns]

        if len(available_risk_cols) >= 2:
            # Normalize the columns for the composite score
            normalized_cols = {}
            for col in available_risk_cols:
                min_val = X[col].min()
                max_val = X[col].max()
                if max_val > min_val:
                    normalized_cols[col] = (X[col] - min_val) / (max_val - min_val)
                else:
                    normalized_cols[col] = X[col] - min_val

            # Create composite risk score
            X_transformed['composite_risk_score'] = sum(normalized_cols.values()) / len(normalized_cols)
            self.new_feature_names_.append('composite_risk_score')

        logger.info(f"Generated {len(self.new_feature_names_)} new features")
        return X_transformed

    def get_feature_names_out(self):
        return self.input_features_ + self.new_feature_names_


def build_feature_engineering_pipeline(numeric_features, categorical_features,
                                       enable_feature_generation=True,
                                       enable_auto_selection=True,
                                       scaling_method="robust",
                                       imputation_strategy="knn"):
    """
    Build the feature engineering pipeline.

    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        enable_feature_generation: Whether to generate new features
        enable_auto_selection: Whether to perform automatic feature selection
        scaling_method: Method for feature scaling
        imputation_strategy: Method for imputing missing values

    Returns:
        Fully configured sklearn pipeline for feature engineering
    """
    # Create transformers for numeric features
    numeric_transformers = []

    # Step 1: Imputation for missing values
    if imputation_strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
    elif imputation_strategy == 'median':
        imputer = SimpleImputer(strategy='median')
    else:
        imputer = SimpleImputer(strategy='mean')

    numeric_transformers.append(('imputer', imputer))

    # Step 2: Feature scaling
    if scaling_method == 'robust':
        scaler = RobustScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'yeo-johnson':
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        scaler = StandardScaler()

    numeric_transformers.append(('scaler', scaler))

    # Create the numeric pipeline
    numeric_pipeline = Pipeline(steps=numeric_transformers)

    # Create transformers for categorical features
    # Split categorical features into high and low cardinality
    high_card_features = []
    low_card_features = []

    # This would typically be determined dynamically based on cardinality
    # Here we just use a simple heuristic
    for feature in categorical_features:
        if feature in ['grade', 'sub_grade', 'addr_state', 'zip_code']:
            high_card_features.append(feature)
        else:
            low_card_features.append(feature)

    # Build the column transformer
    transformers = [
        ('numeric', numeric_pipeline, numeric_features)
    ]

    # Add categorical transformers if any exist
    if high_card_features:
        high_card_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        transformers.append(('high_card_categorical', high_card_pipeline, high_card_features))

    if low_card_features:
        low_card_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
        ])
        transformers.append(('low_card_categorical', low_card_pipeline, low_card_features))

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )

    # Build the complete pipeline
    pipeline_steps = [('preprocessor', preprocessor)]

    # Add feature generation if enabled
    if enable_feature_generation:
        feature_generator = FeatureGenerator(
            enable_ratios=True,
            enable_interactions=True
        )
        pipeline_steps.append(('feature_generator', feature_generator))

    # Add feature selection if enabled
    if enable_auto_selection:
        # Use variance threshold to remove constant features
        pipeline_steps.append(('variance_threshold', VarianceThreshold(threshold=0.01)))

        # Use feature importance from XGBoost to select the most important features
        feature_selector = SelectFromModel(
            xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4),
            threshold='median'
        )
        pipeline_steps.append(('feature_selector', feature_selector))

    # Create and return the pipeline
    return Pipeline(steps=pipeline_steps)


def go(args):
    """
    Perform feature engineering on the input data.

    Args:
        args: Command line arguments
    """
    # Convert string boolean arguments to actual booleans
    enable_feature_generation = args.enable_feature_generation.lower() == "true"
    enable_auto_selection = args.enable_auto_selection.lower() == "true"

    # Initialize a Weights & Biases run
    run = wandb.init(job_type="feature_engineering")

    logger.info("Downloading input artifact")

    # Download input artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading input data")
    df = pd.read_csv(artifact_local_path)

    logger.info(f"Input data shape: {df.shape}")

    # Identify numeric and categorical features
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target variable from feature lists if present
    target_column = 'loan_status'
    if target_column in numeric_features:
        numeric_features.remove(target_column)

    logger.info(
        f"Identified {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")

    # Split data into features and target
    if target_column in df.columns:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
    else:
        X = df
        y = None

    # Build the feature engineering pipeline
    logger.info("Building feature engineering pipeline")
    pipeline = build_feature_engineering_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        enable_feature_generation=enable_feature_generation,
        enable_auto_selection=enable_auto_selection,
        scaling_method=args.scaling_method,
        imputation_strategy=args.imputation_strategy
    )

    # Fit and transform the data
    logger.info("Applying feature engineering")
    if y is not None and enable_auto_selection:
        X_transformed = pipeline.fit_transform(X, y)
    else:
        X_transformed = pipeline.fit_transform(X)

    logger.info(f"Transformed data shape: {X_transformed.shape}")

    # Convert back to dataframe with appropriate feature names
    # Note: This is complex as the pipeline may not preserve feature names properly
    try:
        feature_names = pipeline.get_feature_names_out()
    except:
        # If feature names not available, use generic names
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # Reattach the target column if it exists
    if y is not None:
        X_transformed_df[target_column] = y.values

    # Save the transformed data
    logger.info("Saving transformed data")
    output_path = "engineered_features.csv"
    X_transformed_df.to_csv(output_path, index=False)

    # Save the pipeline for later use (inference)
    logger.info("Saving feature engineering pipeline")
    pipeline_path = "feature_engineering_pipeline.pkl"
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)

    # Create and upload artifacts
    logger.info(f"Uploading {args.output_artifact} to Weights & Biases")

    # Create data artifact
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(output_path)
    run.log_artifact(artifact)

    # Create pipeline artifact
    pipeline_artifact = wandb.Artifact(
        name=f"{args.output_artifact}_pipeline",
        type="pipeline",
        description="Feature engineering pipeline"
    )
    pipeline_artifact.add_file(pipeline_path)
    run.log_artifact(pipeline_artifact)

    # Log feature counts as metrics
    run.log({
        "n_input_features": X.shape[1],
        "n_engineered_features": X_transformed_df.shape[1] - (1 if y is not None else 0),
        "n_samples": len(X_transformed_df)
    })

    logger.info("Feature engineering complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--enable_feature_generation",
        type=str,
        help="Whether to enable feature generation",
        default="true"
    )

    parser.add_argument(
        "--enable_auto_selection",
        type=str,
        help="Whether to enable automated feature selection",
        default="true"
    )

    parser.add_argument(
        "--scaling_method",
        type=str,
        help="Method for feature scaling",
        default="robust"
    )

    parser.add_argument(
        "--imputation_strategy",
        type=str,
        help="Method for imputing missing values",
        default="knn"
    )

    args = parser.parse_args()

    go(args)