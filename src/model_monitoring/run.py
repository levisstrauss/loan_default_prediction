#!/usr/bin/env python
"""
Model monitoring script for the loan default prediction pipeline.
"""
import argparse
import logging
import pandas as pd
import numpy as np
import os
import pickle
import json
import wandb
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class ModelMonitor:
    """
    Model monitoring system for detecting:
    1. Data drift
    2. Concept drift
    3. Model performance degradation
    4. Feature importance changes
    """

    def __init__(self, model, calibrated_model=None, drift_threshold=0.1,
                 performance_threshold=0.05, feature_importance_threshold=0.2):
        """
        Initialize the model monitor.

        Args:
            model: Trained model
            calibrated_model: Calibrated model (optional)
            drift_threshold: Threshold for drift detection
            performance_threshold: Threshold for performance degradation
            feature_importance_threshold: Threshold for feature importance changes
        """
        self.model = model
        self.calibrated_model = calibrated_model
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.feature_importance_threshold = feature_importance_threshold

        # Initialize monitoring storage
        self.reference_data = None
        self.reference_stats = None
        self.reference_performance = None
        self.reference_feature_importance = None

        # Create monitoring directories
        os.makedirs("monitoring", exist_ok=True)
        os.makedirs("monitoring/plots", exist_ok=True)

    def set_reference(self, X_ref, y_ref=None, feature_importance=None, performance_metrics=None):
        """
        Set the reference data for drift detection.

        Args:
            X_ref: Reference features
            y_ref: Reference targets (optional)
            feature_importance: Reference feature importance (optional)
            performance_metrics: Reference performance metrics (optional)
        """
        logger.info("Setting reference data for monitoring")

        # Store reference data
        self.reference_data = {
            'X': X_ref,
            'y': y_ref
        }

        # Calculate reference statistics
        self.reference_stats = self._calculate_data_statistics(X_ref)

        # Store reference feature importance
        if feature_importance is not None:
            self.reference_feature_importance = feature_importance.copy()
        else:
            # Calculate feature importance if model is available
            if self.model is not None and X_ref is not None:
                self.reference_feature_importance = self._calculate_feature_importance(X_ref)

        # Store reference performance metrics
        if performance_metrics is not None:
            self.reference_performance = performance_metrics.copy()
        else:
            # Calculate performance metrics if model and labels are available
            if self.model is not None and X_ref is not None and y_ref is not None:
                self.reference_performance = self._calculate_performance_metrics(X_ref, y_ref)

        # Save reference data to disk
        self._save_reference_data()

        logger.info("Reference data set and saved")

    def _calculate_data_statistics(self, X):
        """
        Calculate statistics for each feature.

        Args:
            X: Feature dataframe

        Returns:
            Dictionary with statistics for each feature
        """
        stats = {}

        for column in X.columns:
            col_stats = {}

            # Determine data type
            data_type = X[column].dtype
            col_stats['dtype'] = str(data_type)

            if pd.api.types.is_numeric_dtype(X[column]):
                # Calculate numeric statistics
                col_stats['mean'] = X[column].mean()
                col_stats['median'] = X[column].median()
                col_stats['std'] = X[column].std()
                col_stats['min'] = X[column].min()
                col_stats['max'] = X[column].max()
                col_stats['missing'] = X[column].isna().mean()

                # Calculate percentiles
                for p in [1, 5, 25, 50, 75, 95, 99]:
                    col_stats[f'p{p}'] = X[column].quantile(p / 100)
            else:
                # Calculate categorical statistics
                value_counts = X[column].value_counts(normalize=True)
                col_stats['unique_count'] = len(value_counts)
                col_stats['missing'] = X[column].isna().mean()

                # Store top categories
                top_n = min(10, len(value_counts))
                col_stats['top_categories'] = value_counts.head(top_n).to_dict()

            stats[column] = col_stats

        return stats

    def _calculate_feature_importance(self, X):
        """
        Calculate feature importance using the model.

        Args:
            X: Feature dataframe

        Returns:
            DataFrame with feature importance
        """
        # Get feature importance from model
        dmatrix = xgb.DMatrix(X)
        importance_type = 'gain'
        importance = self.model.get_score(importance_type=importance_type)

        # Convert to DataFrame
        importance_df = pd.DataFrame([importance]).T
        importance_df.reset_index(inplace=True)
        importance_df.columns = ['Feature', 'Importance']

        # Add missing features with zero importance
        missing_features = set(X.columns) - set(importance_df['Feature'])
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

    def _calculate_performance_metrics(self, X, y):
        """
        Calculate performance metrics for the model.

        Args:
            X: Feature dataframe
            y: Target series

        Returns:
            Dictionary with performance metrics
        """
        # Get predictions
        dmatrix = xgb.DMatrix(X)
        raw_probs = self.model.predict(dmatrix)

        # Use calibrated model if available
        if self.calibrated_model is not None:
            probs = self.calibrated_model.predict_proba(X)[:, 1]
        else:
            probs = raw_probs

        # Convert to binary predictions
        y_pred = (probs >= 0.5).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred)),
            'recall': float(recall_score(y, y_pred)),
            'f1': float(f1_score(y, y_pred)),
            'auc': float(roc_auc_score(y, probs))
        }

        return metrics

    def _save_reference_data(self):
        """
        Save reference data to disk.
        """
        # Save reference statistics
        if self.reference_stats is not None:
            with open('monitoring/reference_stats.json', 'w') as f:
                json.dump(self.reference_stats, f, indent=2, default=str)

        # Save reference feature importance
        if self.reference_feature_importance is not None:
            self.reference_feature_importance.to_csv('monitoring/reference_feature_importance.csv', index=False)

        # Save reference performance metrics
        if self.reference_performance is not None:
            with open('monitoring/reference_performance.json', 'w') as f:
                json.dump(self.reference_performance, f, indent=2)

    def detect_data_drift(self, X_current):
        """
        Detect drift in feature distributions.

        Args:
            X_current: Current feature dataframe

        Returns:
            Dictionary with drift detection results
        """
        if self.reference_stats is None:
            logger.warning("No reference statistics available for drift detection")
            return {
                'has_drift': False,
                'drift_score': 0,
                'drifted_features': []
            }

        logger.info("Detecting data drift")

        # Calculate current statistics
        current_stats = self._calculate_data_statistics(X_current)

        # Compare statistics for each feature
        feature_drift_scores = {}
        drifted_features = []

        for feature in self.reference_stats.keys():
            if feature not in X_current.columns:
                logger.warning(f"Feature {feature} not found in current data")
                continue

            ref_stats = self.reference_stats[feature]
            cur_stats = current_stats[feature]

            # Check data type compatibility
            if ref_stats['dtype'] != cur_stats['dtype']:
                logger.warning(f"Data type mismatch for feature {feature}")
                feature_drift_scores[feature] = 1.0  # Maximum drift
                drifted_features.append(feature)
                continue

            # Calculate drift score
            if 'mean' in ref_stats:  # Numeric feature
                # Numeric drift detection

                # Use Wasserstein distance for numeric features
                if self.reference_data is not None and 'X' in self.reference_data:
                    # Get reference and current values
                    ref_values = self.reference_data['X'][feature].dropna().values
                    cur_values = X_current[feature].dropna().values

                    # Calculate Wasserstein distance (Earth Mover's Distance)
                    try:
                        w_dist = wasserstein_distance(ref_values, cur_values)
                        # Normalize by range
                        range_val = ref_stats['max'] - ref_stats['min']
                        if range_val > 0:
                            w_dist_norm = w_dist / range_val
                        else:
                            w_dist_norm = 0

                        # Calculate KS statistic
                        ks_stat, _ = ks_2samp(ref_values, cur_values)

                        # Combine distances for final score (higher weight to KS test)
                        drift_score = 0.4 * w_dist_norm + 0.6 * ks_stat

                    except Exception as e:
                        logger.warning(f"Error calculating drift for {feature}: {e}")
                        drift_score = 0
                else:
                    # Fallback to comparing statistics
                    mean_diff = abs(cur_stats['mean'] - ref_stats['mean'])
                    if ref_stats['std'] > 0:
                        mean_diff_norm = mean_diff / ref_stats['std']
                    else:
                        mean_diff_norm = mean_diff

                    std_diff = abs(cur_stats['std'] - ref_stats['std'])
                    if ref_stats['std'] > 0:
                        std_diff_norm = std_diff / ref_stats['std']
                    else:
                        std_diff_norm = std_diff

                    # Combine for final score
                    drift_score = 0.7 * mean_diff_norm + 0.3 * std_diff_norm
            else:
                # Categorical drift detection

                # Calculate distribution difference
                ref_categories = ref_stats.get('top_categories', {})
                cur_categories = cur_stats.get('top_categories', {})

                all_categories = set(ref_categories.keys()) | set(cur_categories.keys())
                total_diff = 0

                for category in all_categories:
                    ref_freq = ref_categories.get(category, 0)
                    cur_freq = cur_categories.get(category, 0)
                    total_diff += abs(ref_freq - cur_freq)

                # Normalize
                drift_score = total_diff / 2  # Sum of abs diffs is at most 2

            # Cap at 1.0
            drift_score = min(drift_score, 1.0)

            # Store drift score
            feature_drift_scores[feature] = drift_score

            # Check if drift exceeds threshold
            if drift_score > self.drift_threshold:
                drifted_features.append(feature)

        # Calculate overall drift score
        if feature_drift_scores:
            # Use feature importance as weights if available
            if self.reference_feature_importance is not None:
                # Create dictionary mapping feature to importance
                importance_dict = dict(zip(
                    self.reference_feature_importance['Feature'],
                    self.reference_feature_importance['Relative_Importance']
                ))

                weighted_scores = []
                total_weight = 0

                for feature, score in feature_drift_scores.items():
                    weight = importance_dict.get(feature, 0)
                    weighted_scores.append(score * weight)
                    total_weight += weight

                if total_weight > 0:
                    overall_drift_score = sum(weighted_scores) / total_weight
                else:
                    overall_drift_score = sum(feature_drift_scores.values()) / len(feature_drift_scores)
            else:
                overall_drift_score = sum(feature_drift_scores.values()) / len(feature_drift_scores)
        else:
            overall_drift_score = 0

        # Plot top drifted features
        if drifted_features:
            self._plot_drifted_features(X_current, feature_drift_scores, drifted_features)

        return {
            'has_drift': len(drifted_features) > 0,
            'drift_score': overall_drift_score,
            'drifted_features': drifted_features,
            'feature_drift_scores': feature_drift_scores
        }

    def detect_concept_drift(self, X_current, y_current=None):
        """
        Detect concept drift by comparing model performance.

        Args:
            X_current: Current features
            y_current: Current target (optional)

        Returns:
            Dictionary with concept drift detection results
        """
        # If no reference performance data is available, we can't detect concept drift
        if self.reference_performance is None:
            logger.warning("No reference performance metrics available for concept drift detection")
            return {
                'has_drift': False,
                'drift_score': 0
            }

        # If no current labels are provided, we can't detect concept drift
        if y_current is None:
            logger.warning("No target values provided for concept drift detection")
            return {
                'has_drift': False,
                'drift_score': 0
            }

        logger.info("Detecting concept drift")

        # Calculate current performance metrics
        current_metrics = self._calculate_performance_metrics(X_current, y_current)

        # Calculate performance differences
        perf_diff = {}
        for metric in self.reference_performance:
            if metric in current_metrics:
                # Calculate relative difference
                if self.reference_performance[metric] != 0:
                    rel_diff = (current_metrics[metric] - self.reference_performance[metric]) / abs(
                        self.reference_performance[metric])
                else:
                    rel_diff = current_metrics[metric]

                perf_diff[metric] = rel_diff

        # Calculate concept drift score based on AUC decrease
        if 'auc' in perf_diff:
            # Negative rel_diff means AUC decreased
            concept_drift_score = max(0, -perf_diff['auc'])
        else:
            # Use average of precision and recall if AUC not available
            pr_diff = perf_diff.get('precision', 0)
            re_diff = perf_diff.get('recall', 0)
            concept_drift_score = max(0, -(pr_diff + re_diff) / 2)

        # Check if concept drift exceeds threshold
        has_drift = concept_drift_score > self.drift_threshold

        # Plot performance comparison
        self._plot_performance_comparison(self.reference_performance, current_metrics)

        return {
            'has_drift': has_drift,
            'drift_score': concept_drift_score,
            'performance_diff': perf_diff,
            'current_metrics': current_metrics,
            'reference_metrics': self.reference_performance
        }

    def detect_feature_importance_drift(self, X_current):
        """
        Detect drift in feature importance.

        Args:
            X_current: Current features

        Returns:
            Dictionary with feature importance drift detection results
        """
        if self.reference_feature_importance is None:
            logger.warning("No reference feature importance available")
            return {
                'has_drift': False,
                'importance_changes': {},
                'changed_features': []
            }

        logger.info("Detecting feature importance drift")

        # Calculate current feature importance
        current_importance = self._calculate_feature_importance(X_current)

        # Merge reference and current importance
        importance_comparison = pd.merge(
            self.reference_feature_importance[['Feature', 'Relative_Importance']],
            current_importance[['Feature', 'Relative_Importance']],
            on='Feature',
            how='outer',
            suffixes=('_ref', '_cur')
        ).fillna(0)

        # Calculate absolute and relative changes
        importance_comparison['absolute_change'] = importance_comparison['Relative_Importance_cur'] - \
                                                   importance_comparison['Relative_Importance_ref']

        # Avoid division by zero
        importance_comparison['relative_change'] = importance_comparison['absolute_change'] / importance_comparison[
            'Relative_Importance_ref'].clip(lower=0.1)

        # Identify features with significant changes
        changed_features = importance_comparison[
            abs(importance_comparison['relative_change']) > self.feature_importance_threshold
            ]['Feature'].tolist()

        # Plot feature importance changes
        self._plot_feature_importance_changes(importance_comparison)

        return {
            'has_drift': len(changed_features) > 0,
            'importance_changes': importance_comparison.to_dict(orient='records'),
            'changed_features': changed_features
        }

    def _plot_drifted_features(self, X_current, feature_drift_scores, drifted_features, max_features=5):
        """
        Plot distribution of top drifted features.

        Args:
            X_current: Current features
            feature_drift_scores: Dictionary of drift scores by feature
            drifted_features: List of drifted features
            max_features: Maximum number of features to plot
        """
        # Sort drifted features by drift score
        sorted_features = sorted(
            [(f, feature_drift_scores[f]) for f in drifted_features],
            key=lambda x: x[1],
            reverse=True
        )

        # Select top features to plot
        top_features = sorted_features[:max_features]

        if not top_features:
            return

        # Create a figure for each drifted feature
        for feature, drift_score in top_features:
            plt.figure(figsize=(10, 6))

            # Get reference data
            if self.reference_data is not None and 'X' in self.reference_data:
                ref_values = self.reference_data['X'][feature].dropna()

                # Check if numeric or categorical
                if pd.api.types.is_numeric_dtype(ref_values):
                    # Numeric feature - plot histogram
                    plt.hist(ref_values, alpha=0.5, label='Reference', bins=30, color='blue')
                    plt.hist(X_current[feature].dropna(), alpha=0.5, label='Current', bins=30, color='red')
                    plt.xlabel(feature)
                    plt.ylabel('Frequency')
                else:
                    # Categorical feature - plot bar chart
                    ref_counts = ref_values.value_counts(normalize=True)
                    cur_counts = X_current[feature].value_counts(normalize=True)

                    # Get all categories
                    all_cats = sorted(list(set(ref_counts.index) | set(cur_counts.index)))

                    # Limit to top 10 categories
                    if len(all_cats) > 10:
                        # Use the top categories from reference data
                        top_cats = set(ref_counts.nlargest(10).index)
                        # Add any current top categories
                        top_cats.update(cur_counts.nlargest(10).index)
                        all_cats = sorted(list(top_cats))

                    # Get values for each category
                    ref_values = [ref_counts.get(cat, 0) for cat in all_cats]
                    cur_values = [cur_counts.get(cat, 0) for cat in all_cats]

                    # Set up bar positions
                    x = np.arange(len(all_cats))
                    width = 0.35

                    # Create bar chart
                    plt.bar(x - width / 2, ref_values, width, label='Reference', color='blue')
                    plt.bar(x + width / 2, cur_values, width, label='Current', color='red')
                    plt.xticks(x, all_cats, rotation=45, ha='right')
                    plt.xlabel('Category')
                    plt.ylabel('Frequency')

            plt.title(f"{feature} Distribution (Drift Score: {drift_score:.4f})")
            plt.legend()
            plt.tight_layout()

            # Save plot
            plt.savefig(f"monitoring/plots/drift_{feature}.png")
            plt.close()

    def _plot_performance_comparison(self, ref_metrics, cur_metrics):
        """
        Plot performance metric comparison.

        Args:
            ref_metrics: Reference performance metrics
            cur_metrics: Current performance metrics
        """
        # Get common metrics
        common_metrics = sorted(set(ref_metrics.keys()) & set(cur_metrics.keys()))

        if not common_metrics:
            return

        # Create bar chart
        plt.figure(figsize=(10, 6))

        x = np.arange(len(common_metrics))
        width = 0.35

        ref_values = [ref_metrics[m] for m in common_metrics]
        cur_values = [cur_metrics[m] for m in common_metrics]

        plt.bar(x - width / 2, ref_values, width, label='Reference', color='blue')
        plt.bar(x + width / 2, cur_values, width, label='Current', color='red')

        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Performance Metric Comparison')
        plt.xticks(x, common_metrics)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plt.savefig(f"monitoring/plots/performance_comparison.png")
        plt.close()

    def _plot_feature_importance_changes(self, importance_comparison, top_n=10):
        """
        Plot feature importance changes.

        Args:
            importance_comparison: DataFrame with feature importance comparison
            top_n: Number of top features to display
        """
        # Sort by absolute change
        sorted_comparison = importance_comparison.sort_values('absolute_change', key=abs, ascending=False)

        # Select top features
        top_features = sorted_comparison.head(top_n)

        if len(top_features) == 0:
            return

        # Create bar chart
        plt.figure(figsize=(12, 8))

        plt.barh(top_features['Feature'], top_features['Relative_Importance_ref'],
                 alpha=0.5, color='blue', label='Reference')
        plt.barh(top_features['Feature'], top_features['Relative_Importance_cur'],
                 alpha=0.5, color='red', label='Current')

        plt.xlabel('Relative Importance (%)')
        plt.ylabel('Feature')
        plt.title('Feature Importance Changes')
        plt.legend()
        plt.tight_layout()

        # Save plot
        plt.savefig(f"monitoring/plots/feature_importance_changes.png")
        plt.close()

        # Also plot the absolute changes
        plt.figure(figsize=(12, 8))
        colors = ['red' if x > 0 else 'blue' for x in top_features['absolute_change']]
        plt.barh(top_features['Feature'], top_features['absolute_change'], color=colors)
        plt.axvline(x=0, color='black', linestyle='-')
        plt.xlabel('Absolute Change in Importance (%)')
        plt.ylabel('Feature')
        plt.title('Feature Importance Changes (Absolute)')
        plt.tight_layout()

        # Save plot
        plt.savefig(f"monitoring/plots/feature_importance_changes_abs.png")
        plt.close()


def go(args):
    """
    Run the model monitoring system.

    Args:
        args: Command line arguments
    """
    # Initialize a Weights & Biases run
    run = wandb.init(job_type="model_monitoring")

    # Create monitoring directory
    os.makedirs("monitoring", exist_ok=True)

    # Download the model artifact
    logger.info(f"Loading model from {args.model}")
    model_artifact = run.use_artifact(args.model)
    model_dir = model_artifact.download()

    # Load XGBoost model
    model_path = os.path.join(model_dir, "best_model.json")
    model = xgb.Booster()
    model.load_model(model_path)

    # Try to load calibrated model if it exists
    calibrated_model_path = os.path.join(model_dir, "calibrated_model.pkl")
    calibrated_model = None
    if os.path.exists(calibrated_model_path):
        logger.info("Loading calibrated model")
        with open(calibrated_model_path, 'rb') as f:
            calibrated_model = pickle.load(f)

    # Load reference data
    logger.info(f"Loading reference data from {args.reference_data}")
    ref_data_artifact = run.use_artifact(args.reference_data)
    ref_data_path = ref_data_artifact.file()

    ref_df = pd.read_csv(ref_data_path)
    logger.info(f"Reference data shape: {ref_df.shape}")

    # Split features and target
    X_ref = ref_df.drop(args.target_column, axis=1)
    y_ref = ref_df[args.target_column]

    # Initialize the model monitor
    logger.info("Initializing model monitor")
    monitor = ModelMonitor(
        model=model,
        calibrated_model=calibrated_model,
        drift_threshold=args.drift_threshold,
        performance_threshold=args.performance_threshold,
        feature_importance_threshold=args.feature_importance_threshold
    )

    # Load feature importance if available
    feature_importance_path = os.path.join(model_dir, "feature_importance.csv")
    feature_importance = None
    if os.path.exists(feature_importance_path):
        logger.info("Loading feature importance")
        feature_importance = pd.read_csv(feature_importance_path)

    # Set reference data
    logger.info("Setting reference data")
    monitor.set_reference(X_ref, y_ref, feature_importance)

    # Just as an example, we'll use the same data as current data
    # In a real scenario, this would be new data
    X_current = X_ref
    y_current = y_ref

    # Detect data drift
    logger.info("Detecting data drift")
    drift_results = monitor.detect_data_drift(X_current)

    # Detect concept drift
    logger.info("Detecting concept drift")
    concept_drift_results = monitor.detect_concept_drift(X_current, y_current)

    # Detect feature importance drift
    logger.info("Detecting feature importance drift")
    importance_drift_results = monitor.detect_feature_importance_drift(X_current)

    # Combine results
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_drift': drift_results,
        'concept_drift': concept_drift_results,
        'feature_importance_drift': importance_drift_results
    }

    # Save results
    with open('monitoring/monitoring_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Create a summary dashboard
    logger.info("Creating monitoring dashboard")
    monitor._create_dashboard(results)

    # Log monitoring results to W&B
    logger.info("Logging results to W&B")

    # Log metrics
    wandb.log({
        'data_drift_score': drift_results['drift_score'],
        'concept_drift_score': concept_drift_results['drift_score'],
        'drifted_features_count': len(drift_results['drifted_features']),
        'changed_importance_features_count': len(importance_drift_results['changed_features'])
    })

    # Create monitoring artifact
    monitoring_artifact = wandb.Artifact(
        name="model_monitoring_results",
        type="model_monitoring",
        description="Model monitoring results and visualizations"
    )

    # Add monitoring results
    monitoring_artifact.add_file('monitoring/monitoring_results.json')

    # Add monitoring plots
    for plot_file in os.listdir('monitoring/plots'):
        if plot_file.endswith('.png'):
            monitoring_artifact.add_file(os.path.join('monitoring/plots', plot_file))

    # Add monitoring dashboard
    if os.path.exists('monitoring/dashboard.html'):
        monitoring_artifact.add_file('monitoring/dashboard.html')

    # Log the artifact
    run.log_artifact(monitoring_artifact)

    # Print summary
    logger.info(f"Data drift score: {drift_results['drift_score']:.4f}")
    logger.info(f"Concept drift score: {concept_drift_results['drift_score']:.4f}")
    logger.info(f"Drifted features: {len(drift_results['drifted_features'])}")
    logger.info(f"Changed importance features: {len(importance_drift_results['changed_features'])}")

    # Alert if any drift detected
    if drift_results['has_drift'] or concept_drift_results['has_drift'] or importance_drift_results['has_drift']:
        logger.warning("ALERT: Drift detected in the model!")
        if drift_results['has_drift']:
            logger.warning(f"Data drift detected in features: {drift_results['drifted_features']}")
        if concept_drift_results['has_drift']:
            logger.warning(f"Concept drift detected with score: {concept_drift_results['drift_score']:.4f}")
        if importance_drift_results['has_drift']:
            logger.warning(
                f"Feature importance drift detected in features: {importance_drift_results['changed_features']}")


def _create_dashboard(self, results):
    """
    Create a simple HTML dashboard to visualize monitoring results.

    Args:
        results: Dictionary of monitoring results
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import base64

        # Create a subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Data Drift", "Concept Drift",
                "Feature Importance Changes", "Performance Metrics"
            ],
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )

        # Add data drift gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results['data_drift']['drift_score'],
                title={"text": "Data Drift Score"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, self.drift_threshold], "color": "lightgreen"},
                        {"range": [self.drift_threshold, 1], "color": "red"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": self.drift_threshold
                    }
                }
            ),
            row=1, col=1
        )

        # Add concept drift gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results['concept_drift']['drift_score'],
                title={"text": "Concept Drift Score"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, self.drift_threshold], "color": "lightgreen"},
                        {"range": [self.drift_threshold, 1], "color": "red"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": self.drift_threshold
                    }
                }
            ),
            row=1, col=2
        )

        # Add feature importance changes
        if 'feature_importance_drift' in results and 'importance_changes' in results['feature_importance_drift']:
            changes = results['feature_importance_drift']['importance_changes']

            if changes and len(changes) > 0:
                # Convert to DataFrame
                changes_df = pd.DataFrame(changes)

                # Sort by absolute change
                sorted_df = changes_df.sort_values('absolute_change', key=abs, ascending=False).head(10)

                # Add bar chart
                fig.add_trace(
                    go.Bar(
                        x=sorted_df['absolute_change'],
                        y=sorted_df['Feature'],
                        orientation='h',
                        marker_color=['red' if x > 0 else 'blue' for x in sorted_df['absolute_change']]
                    ),
                    row=2, col=1
                )

                # Update layout
                fig.update_xaxes(title_text="Importance Change", row=2, col=1)
                fig.update_yaxes(title_text="Feature", row=2, col=1)

        # Add performance metrics
        if 'concept_drift' in results and 'current_metrics' in results['concept_drift']:
            cur_metrics = results['concept_drift']['current_metrics']
            ref_metrics = results['concept_drift']['reference_metrics']

            if cur_metrics and ref_metrics:
                # Get common metrics
                common_metrics = sorted(set(cur_metrics.keys()) & set(ref_metrics.keys()))

                if common_metrics:
                    # Add bar chart
                    fig.add_trace(
                        go.Bar(
                            x=common_metrics,
                            y=[cur_metrics[m] for m in common_metrics],
                            name='Current',
                            marker_color='red'
                        ),
                        row=2, col=2
                    )

                    fig.add_trace(
                        go.Bar(
                            x=common_metrics,
                            y=[ref_metrics[m] for m in common_metrics],
                            name='Reference',
                            marker_color='blue'
                        ),
                        row=2, col=2
                    )

                    # Update layout
                    fig.update_xaxes(title_text="Metric", row=2, col=2)
                    fig.update_yaxes(title_text="Value", row=2, col=2)

        # Update layout
        fig.update_layout(
            title_text="Model Monitoring Dashboard",
            height=800,
            width=1000,
            showlegend=True
        )

        # Save as HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Monitoring Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                .summary {{
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-left: 5px solid #007bff;
                }}
                .alert {{
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f8d7da;
                    border-left: 5px solid #dc3545;
                    color: #721c24;
                }}
                .info {{
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #d1ecf1;
                    border-left: 5px solid #17a2b8;
                    color: #0c5460;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                table, th, td {{
                    border: 1px solid #ddd;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Monitoring Dashboard</h1>

                <div class="summary">
                    <h2>Monitoring Summary</h2>
                    <p><strong>Timestamp:</strong> {results.get('timestamp', 'N/A')}</p>
                    <p><strong>Data Drift Score:</strong> {results['data_drift']['drift_score']:.4f}</p>
                    <p><strong>Concept Drift Score:</strong> {results['concept_drift']['drift_score']:.4f}</p>
                    <p><strong>Drifted Features:</strong> {len(results['data_drift']['drifted_features'])}</p>
                    <p><strong>Changed Importance Features:</strong> {len(results['feature_importance_drift']['changed_features'])}</p>
                </div>
        """

        # Add alerts if drift detected
        if (results['data_drift']['has_drift'] or
                results['concept_drift']['has_drift'] or
                results['feature_importance_drift']['has_drift']):

            html_content += """
                <div class="alert">
                    <h2>⚠️ Drift Alerts</h2>
            """

            if results['data_drift']['has_drift']:
                html_content += f"""
                    <p><strong>Data Drift Alert:</strong> Significant data drift detected in 
                    {len(results['data_drift']['drifted_features'])} features.</p>
                """

            if results['concept_drift']['has_drift']:
                html_content += f"""
                    <p><strong>Concept Drift Alert:</strong> Concept drift detected with score 
                    {results['concept_drift']['drift_score']:.4f}.</p>
                """

            if results['feature_importance_drift']['has_drift']:
                html_content += f"""
                    <p><strong>Feature Importance Drift Alert:</strong> Feature importance changes detected in 
                    {len(results['feature_importance_drift']['changed_features'])} features.</p>
                """

            html_content += """
                </div>
            """

        # Add plotly figure
        html_content += f"""
                <div id="plotly-dashboard">
                    {fig.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
        """

        # Add drifted features table if any
        if results['data_drift']['drifted_features']:
            html_content += """
                <div class="info">
                    <h2>Drifted Features</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Drift Score</th>
                        </tr>
            """

            for feature in results['data_drift']['drifted_features']:
                score = results['data_drift']['feature_drift_scores'].get(feature, 'N/A')
                if score != 'N/A':
                    score = f"{score:.4f}"

                html_content += f"""
                        <tr>
                            <td>{feature}</td>
                            <td>{score}</td>
                        </tr>
                """

            html_content += """
                    </table>
                </div>
            """

        # Add performance metrics table
        if 'concept_drift' in results and 'current_metrics' in results['concept_drift']:
            cur_metrics = results['concept_drift']['current_metrics']
            ref_metrics = results['concept_drift']['reference_metrics']

            if cur_metrics and ref_metrics:
                html_content += """
                    <div class="info">
                        <h2>Performance Metrics</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Current</th>
                                <th>Reference</th>
                                <th>Difference</th>
                            </tr>
                """

                for metric in sorted(set(cur_metrics.keys()) & set(ref_metrics.keys())):
                    cur_val = cur_metrics[metric]
                    ref_val = ref_metrics[metric]
                    diff = cur_val - ref_val

                    html_content += f"""
                            <tr>
                                <td>{metric}</td>
                                <td>{cur_val:.4f}</td>
                                <td>{ref_val:.4f}</td>
                                <td>{diff:.4f}</td>
                            </tr>
                    """

                html_content += """
                        </table>
                    </div>
                """

        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """

        # Save HTML dashboard
        with open('monitoring/dashboard.html', 'w') as f:
            f.write(html_content)

        logger.info("Dashboard created successfully")

    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model monitoring")

    parser.add_argument(
        "--model",
        type=str,
        help="Model artifact",
        required=True
    )

    parser.add_argument(
        "--reference_data",
        type=str,
        help="Reference data artifact for drift detection",
        required=True
    )

    parser.add_argument(
        "--drift_threshold",
        type=float,
        help="Threshold for drift detection",
        default=0.1
    )

    parser.add_argument(
        "--performance_threshold",
        type=float,
        help="Threshold for performance degradation",
        default=0.05
    )

    parser.add_argument(
        "--feature_importance_threshold",
        type=float,
        help="Threshold for feature importance changes",
        default=0.2
    )

    parser.add_argument(
        "--target_column",
        type=str,
        help="Target column name",
        default="loan_status"
    )

    args = parser.parse_args()

    go(args)