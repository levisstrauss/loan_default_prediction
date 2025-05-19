#!/usr/bin/env python
"""
Model training script for the loan default prediction pipeline.
"""
import argparse
import logging
import pandas as pd
import numpy as np
import json
import os
import wandb
import mlflow
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def objective(trial, X_train, y_train, X_val, y_val, base_params, optimize_for='auc'):
    """
    Optuna objective function for XGBoost hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        base_params: Base XGBoost parameters
        optimize_for: Metric to optimize for

    Returns:
        Optimization score (higher is better for auc, f1, etc.)
    """
    # Define hyperparameter search space
    params = base_params.copy()

    # Add hyperparameters to search
    params.update({
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True)
    })

    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train with early stopping
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False,
        evals_result=evals_result
    )

    # Make predictions
    preds = model.predict(dval)

    # Calculate the metric to optimize
    if optimize_for == 'auc':
        score = roc_auc_score(y_val, preds)
    elif optimize_for == 'f1':
        y_pred = (preds > 0.5).astype(int)
        score = f1_score(y_val, y_pred)
    elif optimize_for == 'precision':
        y_pred = (preds > 0.5).astype(int)
        score = precision_score(y_val, y_pred)
    elif optimize_for == 'recall':
        y_pred = (preds > 0.5).astype(int)
        score = recall_score(y_val, y_pred)
    elif optimize_for == 'accuracy':
        y_pred = (preds > 0.5).astype(int)
        score = accuracy_score(y_val, y_pred)
    else:
        raise ValueError(f"Unknown optimization metric: {optimize_for}")

    return score


def train_best_model(X_train, y_train, X_val, y_val, best_params, early_stopping_rounds=50):
    """
    Train the best model with the optimized hyperparameters.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        best_params: Best hyperparameters from optimization
        early_stopping_rounds: Early stopping patience

    Returns:
        Trained XGBoost model
    """
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train the model
    evals_result = {}
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100,
        evals_result=evals_result
    )

    return model, evals_result


def calibrate_model(model, X_train, y_train, X_val, y_val, method='isotonic'):
    """
    Calibrate the model to produce accurate probability estimates.

    Args:
        model: Trained XGBoost model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        method: Calibration method ('isotonic' or 'sigmoid')

    Returns:
        Calibrated model
    """
    logger.info(f"Calibrating model using {method} method")

    # Create a wrapper for XGBoost that implements scikit-learn's interface
    class XGBWrapper:
        def __init__(self, model):
            self.model = model

        def predict_proba(self, X):
            dX = xgb.DMatrix(X)
            preds = self.model.predict(dX)
            return np.vstack([1 - preds, preds]).T

    # Create the XGBoost wrapper
    xgb_wrapper = XGBWrapper(model)

    # Apply calibration
    calibrated_model = CalibratedClassifierCV(
        xgb_wrapper,
        method=method,
        cv='prefit'  # Use prefit since we're providing an already trained model
    )

    # Fit the calibration on the validation set
    calibrated_model.fit(X_val, y_val)

    return calibrated_model


def evaluate_model(model, calibrated_model, X, y, dataset_name="validation"):
    """
    Evaluate the model and return performance metrics.

    Args:
        model: Trained XGBoost model
        calibrated_model: Calibrated model (or None)
        X: Features
        y: Target values
        dataset_name: Name of the dataset for logging

    Returns:
        Dictionary of performance metrics
    """
    logger.info(f"Evaluating model on {dataset_name} dataset")

    # Get raw predictions
    dX = xgb.DMatrix(X)
    raw_preds = model.predict(dX)

    # Get calibrated predictions if available
    if calibrated_model is not None:
        cal_probs = calibrated_model.predict_proba(X)[:, 1]
    else:
        cal_probs = raw_preds

    # Binary predictions (threshold = 0.5)
    y_pred = (cal_probs >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, cal_probs),
        'avg_precision': average_precision_score(y, cal_probs)
    }

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Add additional metrics
    metrics.update({
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    })

    # Log metrics
    for metric, value in metrics.items():
        logger.info(f"{dataset_name} {metric}: {value:.4f}")

    return metrics, raw_preds, cal_probs, y_pred


def generate_evaluation_plots(y_true, y_prob, y_pred, metrics, dataset_name="validation"):
    """
    Generate evaluation plots for the model.

    Args:
        y_true: True target values
        y_prob: Predicted probabilities
        y_pred: Binary predictions
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset

    Returns:
        Dictionary with paths to generated plots
    """
    logger.info(f"Generating evaluation plots for {dataset_name} dataset")

    # Create directory for plots
    os.makedirs("plots", exist_ok=True)

    plot_paths = {}

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.colorbar()

    classes = ['Non-default', 'Default']
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    cm_path = f"plots/confusion_matrix_{dataset_name}.png"
    plt.savefig(cm_path)
    plt.close()
    plot_paths['confusion_matrix'] = cm_path

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = f"plots/roc_curve_{dataset_name}.png"
    plt.savefig(roc_path)
    plt.close()
    plot_paths['roc_curve'] = roc_path

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {metrics["avg_precision"]:.3f})')
    plt.axhline(y=sum(y_true) / len(y_true), color='r', linestyle='--',
                label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.legend(loc="best")
    plt.tight_layout()

    pr_path = f"plots/precision_recall_curve_{dataset_name}.png"
    plt.savefig(pr_path)
    plt.close()
    plot_paths['precision_recall_curve'] = pr_path

    # 4. Calibration Curve
    plt.figure(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve - {dataset_name}')
    plt.tight_layout()

    calibration_path = f"plots/calibration_curve_{dataset_name}.png"
    plt.savefig(calibration_path)
    plt.close()
    plot_paths['calibration_curve'] = calibration_path

    # 5. Prediction Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.5, color='blue',
             label='Non-default')
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.5, color='red',
             label='Default')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title(f'Prediction Distribution - {dataset_name}')
    plt.legend()
    plt.tight_layout()

    dist_path = f"plots/prediction_distribution_{dataset_name}.png"
    plt.savefig(dist_path)
    plt.close()
    plot_paths['prediction_distribution'] = dist_path

    return plot_paths


def calculate_feature_importance(model, X):
    """
    Calculate feature importance for the model.

    Args:
        model: Trained XGBoost model
        X: Feature data with column names

    Returns:
        DataFrame with feature importance
    """
    # Get feature names
    feature_names = X.columns.tolist()

    # Get feature importance scores
    importance_type = 'gain'
    scores = model.get_score(importance_type=importance_type)

    # Convert to DataFrame
    importance_df = pd.DataFrame([scores]).T.reset_index()
    importance_df.columns = ['Feature', 'Importance']

    # Add missing features
    missing_features = set(feature_names) - set(importance_df['Feature'])
    if missing_features:
        missing_df = pd.DataFrame({
            'Feature': list(missing_features),
            'Importance': [0] * len(missing_features)
        })
        importance_df = pd.concat([importance_df, missing_df], ignore_index=True)

    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    # Calculate relative importance
    importance_df['Relative_Importance'] = importance_df['Importance'] / importance_df['Importance'].sum() * 100

    return importance_df


def generate_feature_importance_plot(importance_df, top_n=20):
    """
    Generate a feature importance plot.

    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to display

    Returns:
        Path to the saved plot
    """
    # Select top features
    plot_df = importance_df.head(top_n).copy()

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.barh(plot_df['Feature'], plot_df['Relative_Importance'])
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Relative Importance (%)')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()  # Highest at the top
    plt.tight_layout()

    # Save plot
    plot_path = "plots/feature_importance.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def calculate_shap_values(model, X, plot_dir="plots"):
    """
    Calculate SHAP values and generate SHAP plots.

    Args:
        model: Trained XGBoost model
        X: Feature data
        plot_dir: Directory to save plots

    Returns:
        Dictionary with paths to SHAP plots
    """
    logger.info("Calculating SHAP values")

    # Create directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    # Create a sample if dataset is large
    if len(X) > 1000:
        X_sample = X.sample(1000, random_state=42)
    else:
        X_sample = X

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Generate and save plots
    plot_paths = {}

    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    summary_path = f"{plot_dir}/shap_summary.png"
    plt.savefig(summary_path)
    plt.close()
    plot_paths['summary'] = summary_path

    # Bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = f"{plot_dir}/shap_bar.png"
    plt.savefig(bar_path)
    plt.close()
    plot_paths['bar'] = bar_path

    # Dependence plots for top features
    feature_importance = np.abs(shap_values).mean(0)
    feature_idx = np.argsort(-feature_importance)

    for i in range(min(5, len(feature_idx))):
        feature = X_sample.columns[feature_idx[i]]
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(feature_idx[i], shap_values, X_sample, show=False)
        plt.tight_layout()
        dep_path = f"{plot_dir}/shap_dependence_{feature}.png"
        plt.savefig(dep_path)
        plt.close()
        plot_paths[f'dependence_{feature}'] = dep_path

    return plot_paths


def go(args):
    """
    Train and evaluate the XGBoost model.

    Args:
        args: Command line arguments
    """
    # Initialize a Weights & Biases run
    run = wandb.init(job_type="train_model")

    # Also log to MLflow for model registry
    mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
    mlflow.set_experiment("loan_default_prediction")

    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)

    # Load XGBoost configuration
    logger.info(f"Loading XGBoost configuration from {args.xgb_config}")
    with open(args.xgb_config, 'r') as f:
        xgb_params = json.load(f)

    # Override random seed
    xgb_params['random_state'] = args.random_seed

    logger.info("Loading train and validation data")

    # Download artifacts
    train_path = run.use_artifact(args.train_data).file()
    val_path = run.use_artifact(args.val_data).file()

    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Validation data shape: {val_df.shape}")

    # Split features and target
    target_column = 'loan_status'  # Assumed target column

    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]

    X_val = val_df.drop(target_column, axis=1)
    y_val = val_df[target_column]

    # Hyperparameter optimization
    logger.info(f"Starting hyperparameter optimization with {args.n_trials} trials")

    with mlflow.start_run(run_name="hyperparameter_optimization") as run:
        # Create the study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.random_seed)
        )

        # Run optimization
        study.optimize(
            lambda trial: objective(
                trial, X_train, y_train, X_val, y_val,
                xgb_params, args.optimize_for
            ),
            n_trials=args.n_trials
        )

        # Get best parameters
        best_params = xgb_params.copy()
        best_params.update(study.best_params)

        # Log best parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        mlflow.log_metric("best_score", study.best_value)

        # Optuna visualization (if available)
        try:
            import optuna.visualization as vis

            # Optimization history plot
            history_fig = vis.plot_optimization_history(study)
            history_fig.write_image("plots/optuna_history.png")

            # Parameter importance
            param_fig = vis.plot_param_importances(study)
            param_fig.write_image("plots/optuna_param_importance.png")

            # Register artifacts
            mlflow.log_artifact("plots/optuna_history.png")
            mlflow.log_artifact("plots/optuna_param_importance.png")
        except:
            logger.warning("Optuna visualization failed. Continuing without it.")

    # Train best model
    logger.info("Training best model with optimized hyperparameters")

    with mlflow.start_run(run_name="best_model") as run:
        # Log parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Train the model
        best_model, evals_result = train_best_model(
            X_train, y_train, X_val, y_val, best_params
        )

        # Save model
        model_path = "artifacts/best_model.json"
        best_model.save_model(model_path)

        # Calibrate the model
        calibrated_model = calibrate_model(
            best_model, X_train, y_train, X_val, y_val
        )

        # Save the calibrated model
        import pickle
        calibrated_model_path = "artifacts/calibrated_model.pkl"
        with open(calibrated_model_path, 'wb') as f:
            pickle.dump(calibrated_model, f)

        # Evaluate on validation set
        val_metrics, val_raw_preds, val_cal_preds, val_pred = evaluate_model(
            best_model, calibrated_model, X_val, y_val, "validation"
        )

        # Log metrics
        for metric, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric}", value)

        # Generate evaluation plots
        val_plots = generate_evaluation_plots(
            y_val, val_cal_preds, val_pred, val_metrics, "validation"
        )

        # Log plots
        for plot_name, plot_path in val_plots.items():
            mlflow.log_artifact(plot_path)

        # Calculate feature importance
        importance_df = calculate_feature_importance(best_model, X_train)

        # Save feature importance
        importance_path = "artifacts/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        # Generate feature importance plot
        importance_plot_path = generate_feature_importance_plot(importance_df)
        mlflow.log_artifact(importance_plot_path)

        # Calculate SHAP values
        shap_plots = calculate_shap_values(best_model, X_train)

        # Log SHAP plots
        for plot_name, plot_path in shap_plots.items():
            mlflow.log_artifact(plot_path)

        # Log training curves
        if evals_result:
            # Extract learning curves
            train_curve = evals_result['train']['auc']
            val_curve = evals_result['val']['auc']

            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(train_curve, label='Train')
            plt.plot(val_curve, label='Validation')
            plt.xlabel('Boosting Round')
            plt.ylabel('AUC')
            plt.title('Learning Curves')
            plt.legend()
            plt.tight_layout()

            curve_path = "plots/learning_curves.png"
            plt.savefig(curve_path)
            plt.close()

            mlflow.log_artifact(curve_path)

        # Log the XGBoost model
        mlflow.xgboost.log_model(best_model, "xgboost_model")

        # Create model signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, y_train)

        # Create W&B model artifact
        model_artifact = wandb.Artifact(
            name=args.output_artifact,
            type="model",
            description="Trained XGBoost model for loan default prediction"
        )

        # Add files to the artifact
        model_artifact.add_file(model_path)
        model_artifact.add_file(calibrated_model_path)
        model_artifact.add_file(importance_path)

        # Log the artifact
        run.log_artifact(model_artifact)

        # Set aliases for W&B artifact
        model_artifact.aliases.append("latest")
        model_artifact.aliases.append("best")

        # Log model information
        logger.info(f"Best model AUC: {val_metrics['roc_auc']:.4f}")
        logger.info(f"Best model saved to {model_path}")
        logger.info(f"Calibrated model saved to {calibrated_model_path}")
        logger.info(f"Model artifact: {args.output_artifact}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model")

    parser.add_argument(
        "--train_data",
        type=str,
        help="Training data artifact",
        required=True
    )

    parser.add_argument(
        "--val_data",
        type=str,
        help="Validation data artifact",
        required=True
    )

    parser.add_argument(
        "--xgb_config",
        type=str,
        help="XGBoost configuration file",
        required=True
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed",
        required=True
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        help="Number of hyperparameter optimization trials",
        default=50
    )

    parser.add_argument(
        "--optimize_for",
        type=str,
        help="Metric to optimize (auc, f1, precision, recall)",
        default="auc"
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output model artifact",
        required=True
    )

    args = parser.parse_args()

    go(args)