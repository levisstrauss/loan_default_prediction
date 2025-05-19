#!/usr/bin/env python
"""
Download script for the loan default prediction pipeline.
"""
import argparse
import logging
import os
import tempfile
import wandb
import requests
import zipfile
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Download data from URL, process and upload to W&B

    Args:
        args: Command line arguments
    """
    # Initialize a Weights & Biases run
    run = wandb.init(job_type="download")

    logger.info(f"Downloading data from {args.data_url}")

    # Create a temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "data.zip")

        # Download the zip file
        response = requests.get(args.data_url, stream=True)
        response.raise_for_status()  # Will raise an error for bad responses

        # Save the zip file
        with open(zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logger.info("Download complete. Extracting files...")

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Find the CSV file in the extracted contents
        csv_files = list(Path(tmp_dir).glob("*.csv"))

        if not csv_files:
            raise ValueError("No CSV files found in the downloaded archive")

        csv_path = str(csv_files[0])
        logger.info(f"Found CSV file: {os.path.basename(csv_path)}")

        # Load the data to ensure it's valid
        # Skip the first row since it often contains notes in LendingClub data
        df = pd.read_csv(csv_path, skiprows=1)

        # Remove any trailing notes/rows that aren't data
        df = df[df.iloc[:, 0].notnull()]

        logger.info(f"Loaded dataset with shape: {df.shape}")

        # Save the cleaned data as CSV
        output_path = os.path.join(tmp_dir, args.artifact_name)
        df.to_csv(output_path, index=False)

        logger.info(f"Uploading {args.artifact_name} to Weights & Biases")

        # Create and upload the artifact
        artifact = wandb.Artifact(
            name=args.artifact_name,
            type=args.artifact_type,
            description=args.artifact_description
        )
        artifact.add_file(output_path)
        run.log_artifact(artifact)

        logger.info("Upload complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download loan data")

    parser.add_argument(
        "--data_url",
        type=str,
        help="URL to download the data from",
        required=True
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name for the W&B artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the W&B artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the W&B artifact",
        required=True
    )

    args = parser.parse_args()

    go(args)