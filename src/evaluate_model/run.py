#!/usr/bin/env python
"""
Model evaluation script for the loan default prediction pipeline.
"""
import argparse
import logging
import pandas as pd
import numpy as np
import os
import pickle
import json
import wandb
import mlflow
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def evaluate_model(model, calibrated_model, X, y, threshold=0.5):
    """
    Evaluate the model on the given dataset.

    Args:
        model: XGBoost model
        calibrated_model: Calibrated model (or None)
        X: Features
        y: True labels
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Get predictions from raw model
    dmatrix = xgb.DMatrix(X)
    raw_probs = model.predict(dmatrix)

    # Get predictions from calibrated model if available
    if calibrated_model is not None:
        cal_probs = calibrated_model.predict_proba(X)[:, 1]
    else:
        cal_probs = raw_probs

    # Convert probabilities to class predictions
    y_pred = (cal_probs >= threshold).astype(int)

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

    # Calculate additional metrics
    metrics.update({
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False Positive Rate
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    })

    return metrics, raw_probs, cal_probs, y_pred


def generate_evaluation_plots(y_true, y_probs, y_pred, metrics):
    """
    Generate evaluation plots for the model.

    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        y_pred: Predicted labels
        metrics: Dictionary of metrics

    Returns:
        Dictionary of plot paths
    """
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    plot_paths = {}

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
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

    cm_path = "plots/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    plot_paths['confusion_matrix'] = cm_path

    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = "plots/roc_curve.png"
    plt.savefig(roc_path)
    plt.close()
    plot_paths['roc_curve'] = roc_path

    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {metrics["avg_precision"]:.3f})')
    # Add baseline
    baseline = sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='r', linestyle='--',
                label=f'Baseline ({baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.tight_layout()

    pr_path = "plots/precision_recall_curve.png"
    plt.savefig(pr_path)
    plt.close()
    plot_paths['precision_recall_curve'] = pr_path

    # 4. Prediction Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_probs[y_true == 0], bins=30, alpha=0.5, color='blue',
             label='Non-default')
    plt.hist(y_probs[y_true == 1], bins=30, alpha=0.5, color='red',
             label='Default')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Probabilities by Class')
    plt.legend()
    plt.tight_layout()

    dist_path = "plots/prediction_distribution.png"
    plt.savefig(dist_path)
    plt.close()
    plot_paths['prediction_distribution'] = dist_path

    # 5. Threshold Analysis
    thresholds = np.linspace(0.01, 0.99, 50)
    scores = []

    for threshold in thresholds:
        y_pred_t = (y_probs >= threshold).astype(int)
        precision = precision_score(y_true, y_pred_t)
        recall = recall_score(y_true, y_pred_t)
        f1 = f1_score(y_true, y_pred_t)
        accuracy = accuracy_score(y_true, y_pred_t)
        scores.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })

    scores_df = pd.DataFrame(scores)

    plt.figure(figsize=(10, 6))
    plt.plot(scores_df['threshold'], scores_df['precision'], label='Precision')
    plt.plot(scores_df['threshold'], scores_df['recall'], label='Recall')
    plt.plot(scores_df['threshold'], scores_df['f1'], label='F1')
    plt.plot(scores_df['threshold'], scores_df['accuracy'], label='Accuracy')
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default Threshold (0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs. Classification Threshold')
    plt.legend()
    plt.tight_layout()

    threshold_path = "plots/threshold_analysis.png"
    plt.savefig(threshold_path)
    plt.close()
    plot_paths['threshold_analysis'] = threshold_path

    return plot_paths


def go(args):
    """
    Evaluate the model on test data.

    Args:
        args: Command line arguments
    """
    # Initialize a Weights & Biases run
    run = wandb.init(job_type="evaluate_model")

    # Set MLflow tracking
    mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
    mlflow.set_experiment("loan_default_prediction")

    logger.info(f"Loading model from {args.model}")
    model_artifact = run.use_artifact(args.model)
    model_dir = model_artifact.download()

    # Load XGBoost model
    model_path = os.path.join(model_dir, "best_model.json")
    model = xgb.Booster()
    model.load_model(model_path)

    # Try to load calibrated model if it exists
    calibrated_model_path = os.path.join(model_dir, "calibrated_model.pkl")
    if os.path.exists(calibrated_model_path):
        logger.info("Loading calibrated model")
        with open(calibrated_model_path, 'rb') as f:
            calibrated_model = pickle.load(f)
    else:
        logger.info("No calibrated model found")
        calibrated_model = None

    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data_artifact = run.use_artifact(args.test_data)
    test_data_path = test_data_artifact.file()

    test_df = pd.read_csv(test_data_path)
    logger.info(f"Test data shape: {test_df.shape}")

    # Split features and target
    X_test = test_df.drop(args.target_column, axis=1)
    y_test = test_df[args.target_column]

    with mlflow.start_run() as mlflow_run:
        # Evaluate model
        logger.info("Evaluating model on test data")
        metrics, raw_probs, cal_probs, y_pred = evaluate_model(
            model, calibrated_model, X_test, y_test
        )

        # Log metrics to MLflow
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Generate and log plots
        logger.info("Generating evaluation plots")
        plot_paths = generate_evaluation_plots(
            y_test, cal_probs, y_pred, metrics
        )

        # Log plots to MLflow
        for name, path in plot_paths.items():
            mlflow.log_artifact(path)

        # Log classification report
        logger.info("Generating classification report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = "classification_report.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)

        # Create and log evaluation artifact
        eval_artifact = wandb.Artifact(
            name=f"model_evaluation_{run.id}",
            type="model_evaluation",
            description="Evaluation metrics and visualizations for the model"
        )

        # Add all plots to the artifact
        for path in plot_paths.values():
            eval_artifact.add_file(path)

        # Add classification report
        eval_artifact.add_file(report_path)

        # Log evaluation metrics
        metrics_path = "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        eval_artifact.add_file(metrics_path)

        # Log the evaluation artifact
        run.log_artifact(eval_artifact)

        # Log summary metrics to W&B directly
        wandb.log({
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_auc": metrics["roc_auc"],
            "test_avg_precision": metrics["avg_precision"]
        })

        # Print summary
        logger.info(f"Test AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Test F1 Score: {metrics['f1']:.4f}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")

        # If AUC is above 0.80, promote model to "prod" alias
        if metrics["roc_auc"] >= 0.80:
            logger.info(f"Model performance exceeds threshold (AUC >= 0.80)")
            logger.info(f"Adding 'prod' alias to model artifact")
            model_artifact.aliases.append("prod")

            # Also register model in MLflow
            mlflow_model_path = f"mlflow_registry/{run.id}"
            mlflow.xgboost.log_model(
                model,
                mlflow_model_path,
                registered_model_name="loan_default_prediction_model"
            )
            logger.info(f"Model registered in MLflow as 'loan_default_prediction_model'")
        else:
            logger.info(f"Model performance below production threshold (AUC < 0.80)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test data")

    parser.add_argument(
        "--model",
        type=str,
        help="Model artifact",
        required=True
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Test data artifact",
        required=True
    )

    parser.add_argument(
        "--target_column",
        type=str,
        help="Target column name",
        default="loan_status"
    )

    args = parser.parse_args()

    go(args)