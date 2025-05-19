#!/usr/bin/env python
"""
Data split script for the loan default prediction pipeline.
"""
import argparse
import logging
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Split the data into train, validation, and test sets.

    Args:
        args: Command line arguments
    """
    # Initialize a Weights & Biases run
    run = wandb.init(job_type="data_split")

    logger.info("Downloading input artifact")

    # Download input artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading input data")
    df = pd.read_csv(artifact_local_path)

    logger.info(f"Input data shape: {df.shape}")

    # Identify target column (assumed to be 'loan_status')
    target_column = 'loan_status'

    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found in the dataset")

    # Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Determine stratification
    if args.stratify_by != "none" and args.stratify_by in df.columns:
        logger.info(f"Using {args.stratify_by} for stratification")
        stratify = df[args.stratify_by]
    else:
        logger.info("No stratification")
        stratify = None

    # First split: separate test set
    logger.info(f"Splitting data with test_size={args.test_size}, random_state={args.random_seed}")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify
    )

    # Update stratify for the next split
    if stratify is not None:
        stratify = y_train_val

    # Second split: separate validation set from training set
    logger.info(f"Splitting training data with val_size={args.val_size}")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=args.val_size,
        random_state=args.random_seed,
        stratify=stratify
    )

    # Add the target back to create complete dataframes
    train_data = X_train.copy()
    train_data[target_column] = y_train

    val_data = X_val.copy()
    val_data[target_column] = y_val

    test_data = X_test.copy()
    test_data[target_column] = y_test

    # Also create a trainval set (combined train + validation) for model tuning with CV
    trainval_data = pd.concat([train_data, val_data])

    # Log split sizes
    logger.info(f"Train set: {len(train_data)} samples")
    logger.info(f"Validation set: {len(val_data)} samples")
    logger.info(f"Test set: {len(test_data)} samples")
    logger.info(f"Train+Val set: {len(trainval_data)} samples")

    # Save to CSV
    train_path = "train_data.csv"
    val_path = "val_data.csv"
    test_path = "test_data.csv"
    trainval_path = "trainval_data.csv"

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
    trainval_data.to_csv(trainval_path, index=False)

    # Create and upload artifacts

    # Train data artifact
    logger.info("Uploading train data artifact")
    train_artifact = wandb.Artifact(
        name="train_data.csv",
        type="train_data",
        description="Training data split"
    )
    train_artifact.add_file(train_path)
    run.log_artifact(train_artifact)

    # Validation data artifact
    logger.info("Uploading validation data artifact")
    val_artifact = wandb.Artifact(
        name="val_data.csv",
        type="val_data",
        description="Validation data split"
    )
    val_artifact.add_file(val_path)
    run.log_artifact(val_artifact)

    # Test data artifact
    logger.info("Uploading test data artifact")
    test_artifact = wandb.Artifact(
        name="test_data.csv",
        type="test_data",
        description="Test data split"
    )
    test_artifact.add_file(test_path)
    run.log_artifact(test_artifact)

    # Train+Val data artifact (for cross-validation)
    logger.info("Uploading train+val data artifact")
    trainval_artifact = wandb.Artifact(
        name="trainval_data.csv",
        type="trainval_data",
        description="Combined train and validation data"
    )
    trainval_artifact.add_file(trainval_path)
    run.log_artifact(trainval_artifact)

    # Log metrics about the splits
    run.log({
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "trainval_size": len(trainval_data),
        "total_size": len(df),
        "train_pct": len(train_data) / len(df) * 100,
        "val_pct": len(val_data) / len(df) * 100,
        "test_pct": len(test_data) / len(df) * 100
    })

    # Log class distribution in each split
    if target_column in df.columns:
        run.log({
            "train_target_mean": train_data[target_column].mean(),
            "val_target_mean": val_data[target_column].mean(),
            "test_target_mean": test_data[target_column].mean(),
            "total_target_mean": df[target_column].mean()
        })

    logger.info("Data splitting complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train, validation and test sets")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact",
        required=True
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="Size of the test split",
        required=True
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split (from remaining data after test split)",
        required=True
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed",
        required=True
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none"
    )

    args = parser.parse_args()

    go(args)