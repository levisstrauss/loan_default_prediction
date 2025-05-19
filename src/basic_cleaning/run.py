#!/usr/bin/env python
"""
Basic cleaning script for the loan default prediction pipeline.
"""
import argparse
import logging
import pandas as pd
import numpy as np
import wandb
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Clean the data:
    - Remove outliers
    - Handle missing values
    - Convert date columns
    - Fix data types

    Args:
        args: Command line arguments
    """
    # Initialize a Weights & Biases run
    run = wandb.init(job_type="basic_cleaning")

    logger.info("Downloading input artifact")

    # Download the input artifact from W&B
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading input data")
    df = pd.read_csv(artifact_local_path)

    logger.info(f"Original data shape: {df.shape}")

    # Initial data cleaning
    # 1. Remove rows with null target (loan_status)
    if 'loan_status' in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=['loan_status'])
        logger.info(f"Dropped {initial_rows - len(df)} rows with null loan_status")

    # 2. Convert loan_status to binary target
    if 'loan_status' in df.columns:
        df['loan_status'] = df['loan_status'].map(
            lambda x: 1 if x in ['Charged Off', 'Default', 'Late (31-120 days)',
                                 'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off']
            else 0 if x in ['Fully Paid', 'Current', 'Does not meet the credit policy. Status:Fully Paid']
            else np.nan
        )

        # Drop rows with null target after mapping
        df = df.dropna(subset=['loan_status'])
        logger.info(f"Converted loan_status to binary target")

    # 3. Remove outliers based on loan amount
    if 'loan_amnt' in df.columns:
        initial_rows = len(df)
        df = df[df['loan_amnt'].between(args.min_loan_amount, args.max_loan_amount)]
        logger.info(
            f"Dropped {initial_rows - len(df)} rows outside loan amount range: {args.min_loan_amount} - {args.max_loan_amount}")

    # 4. Convert interest rate from string to float
    if 'int_rate' in df.columns and df['int_rate'].dtype == 'object':
        df['int_rate'] = df['int_rate'].str.rstrip('%').astype('float') / 100.0
        logger.info("Converted int_rate from string to float")

    # 5. Convert revol_util from string to float
    if 'revol_util' in df.columns and df['revol_util'].dtype == 'object':
        df['revol_util'] = df['revol_util'].str.rstrip('%').astype('float') / 100.0
        logger.info("Converted revol_util from string to float")

    # 6. Convert term to integer (months)
    if 'term' in df.columns and df['term'].dtype == 'object':
        df['term'] = df['term'].str.strip().str.replace(' months', '').astype('int')
        logger.info("Converted term to integer months")

    # 7. Convert emp_length to numeric years
    if 'emp_length' in df.columns and df['emp_length'].dtype == 'object':
        emp_length_map = {
            '< 1 year': 0,
            '1 year': 1,
            '2 years': 2,
            '3 years': 3,
            '4 years': 4,
            '5 years': 5,
            '6 years': 6,
            '7 years': 7,
            '8 years': 8,
            '9 years': 9,
            '10+ years': 10
        }
        df['emp_length'] = df['emp_length'].map(lambda x: emp_length_map.get(x, np.nan))
        logger.info("Converted emp_length to numeric years")

    # 8. Convert date columns to datetime and create features
    date_columns = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time'])]

    for col in date_columns:
        if pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_datetime64_dtype(df[col]):
            # Try common date formats
            for fmt in ['%b-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%b-%Y']:
                try:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    logger.info(f"Converted {col} to datetime with format {fmt}")
                    break
                except:
                    continue

    # 9. Create additional date features
    if 'issue_d' in df.columns and pd.api.types.is_datetime64_dtype(df['issue_d']):
        df['issue_year'] = df['issue_d'].dt.year
        df['issue_month'] = df['issue_d'].dt.month
        logger.info("Created issue_year and issue_month from issue_d")

    if 'earliest_cr_line' in df.columns and pd.api.types.is_datetime64_dtype(df['earliest_cr_line']):
        today = pd.to_datetime('today')
        df['credit_history_years'] = (today - df['earliest_cr_line']).dt.days / 365.25
        logger.info("Created credit_history_years from earliest_cr_line")

    logger.info(f"Clean data shape: {df.shape}")

    # Save the cleaned data
    logger.info("Saving cleaned data")
    clean_data_path = "clean_data.csv"
    df.to_csv(clean_data_path, index=False)

    logger.info(f"Uploading {args.output_artifact} to Weights & Biases")

    # Create and upload the cleaned data artifact
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(clean_data_path)
    run.log_artifact(artifact)

    logger.info("Basic cleaning complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic data cleaning")

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
        "--min_loan_amount",
        type=float,
        help="Minimum loan amount to include",
        required=True
    )

    parser.add_argument(
        "--max_loan_amount",
        type=float,
        help="Maximum loan amount to include",
        required=True
    )

    args = parser.parse_args()

    go(args)