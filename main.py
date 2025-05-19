import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "feature_engineering",
    "data_split",
    "train_model",
    "evaluate_model",
    "model_monitoring"
]


# This automatically reads in the configuration
@hydra.main(config_path=".", config_name="config")
def go(config: DictConfig):
    """
    Run the loan default prediction pipeline

    Args:
        config (DictConfig): Configuration composed by Hydra
    """
    # Set up the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "download" in active_steps:
            # Download data and load in W&B
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "download"),
                "main",
                parameters={
                    "data_url": config["etl"]["data_url"],
                    "artifact_name": "raw_lending_club_data.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw lending club loan data"
                },
            )

        if "basic_cleaning" in active_steps:
            # Basic data cleaning
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "raw_lending_club_data.csv:latest",
                    "output_artifact": "clean_lending_club_data.csv",
                    "output_type": "clean_data",
                    "output_description": "Cleaned lending club loan data",
                    "min_loan_amount": config['etl']['min_loan_amount'],
                    "max_loan_amount": config['etl']['max_loan_amount']
                },
            )

        if "feature_engineering" in active_steps:
            # Feature engineering
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "feature_engineering"),
                "main",
                parameters={
                    "input_artifact": "clean_lending_club_data.csv:latest",
                    "output_artifact": "features_lending_club_data.csv",
                    "output_type": "features",
                    "output_description": "Feature engineered lending club loan data",
                    "enable_feature_generation": config['feature_engineering']['enable_feature_generation'],
                    "enable_auto_selection": config['feature_engineering']['enable_auto_selection'],
                    "scaling_method": config['feature_engineering']['scaling_method'],
                    "imputation_strategy": config['feature_engineering']['imputation_strategy']
                },
            )

        if "data_split" in active_steps:
            # Split the data into train, validation, and test
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_split"),
                "main",
                parameters={
                    "input_artifact": "features_lending_club_data.csv:latest",
                    "test_size": config['modeling']['test_size'],
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by']
                },
            )

        if "train_model" in active_steps:
            # Serialize the XGBoost configuration into JSON
            xgb_config = os.path.abspath("xgb_config.json")
            with open(xgb_config, "w+") as fp:
                json.dump(dict(config["modeling"]["xgboost"].items()), fp)

            # Train the XGBoost model
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_model"),
                "main",
                parameters={
                    "train_data": "train_data.csv:latest",
                    "val_data": "val_data.csv:latest",
                    "xgb_config": xgb_config,
                    "random_seed": config['modeling']['random_seed'],
                    "n_trials": config['modeling']['optimization']['n_trials'],
                    "optimize_for": config['modeling']['optimization']['optimize_for'],
                    "output_artifact": "xgboost_model"
                },
            )

        if "evaluate_model" in active_steps:
            # Evaluate the model
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "evaluate_model"),
                "main",
                parameters={
                    "model": "xgboost_model:latest",
                    "test_data": "test_data.csv:latest",
                    "target_column": config['etl']['target_column']
                },
            )

        if "model_monitoring" in active_steps:
            # Set up model monitoring
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "model_monitoring"),
                "main",
                parameters={
                    "model": "xgboost_model:latest",
                    "reference_data": "val_data.csv:latest",
                    "drift_threshold": config['monitoring']['drift_threshold'],
                    "performance_threshold": config['monitoring']['performance_threshold'],
                    "feature_importance_threshold": config['monitoring']['feature_importance_threshold'],
                    "target_column": config['etl']['target_column']
                },
            )


if __name__ == "__main__":
    go()