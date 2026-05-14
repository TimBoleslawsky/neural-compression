import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
import argparse
from lightning.pytorch.trainer.states import RunningStage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from framework.utils.factories import (
    dataset_factory,
    model_factory,
    training_module_factory,
    optimizer_factory,
    runner_factory
)


def load_checkpoint_and_data(config, checkpoint_path: str):
    """
    Load trained model and dataset.

    Args:
        config: YAML config
        checkpoint_path: Path to model checkpoint

    Returns:
        runner: Trained model runner
        datamodule: Dataset module with test splits
    """

    # Create dataset
    datamodule = dataset_factory(
        name=config.dataset.name,
        arguments=config.dataset.arguments,
        dataloader_arguments=config.dataloader
    )

    # Setup data split
    datamodule.setup('test')

    # Reconstruct model from config (pass dataset config for shared parameters)
    print("Reconstructing model from config...")
    model = model_factory(
        name=config.model.name,
        arguments=config.model.arguments,
        dataset_config=config.dataset
    )

    training_module = training_module_factory(
        name=config.training_module.name,
        model=model,
        parameters=config.training_module.parameters,
        model_config=config.model,
        full_config=config  # Pass full config for task-aware modules
    )

    optimizer = optimizer_factory(
        name=config.optimizer.name,
        parameters=config.optimizer.parameters
    )

    runner = runner_factory(
        name=config.runner.name,
        parameters={
            'training_module': training_module,
            'optimizer': optimizer,
            **config.runner.parameters
        }
    )

    # Load trained weights
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    runner.load_state_dict(checkpoint['state_dict'])

    # Set model to evaluation mode (not eval() function, but pytorch .eval())
    runner.training_module.model.eval()

    return runner, datamodule


def evaluate_full_dataset(runner, dataloader, stride, window_size, datamodule=None):
    """
    Evaluate model on entire dataset.
    """
    print("\n" + "="*60)
    print("EVALUATING FULL DATASET")
    print("="*60)

    total_samples = len(dataloader.dataset)
    print(f"\nEvaluating {total_samples} samples from test set...")

    # Collect predictions
    all_inputs = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get predictions
            pred, _, _, _ = runner.training_module(batch, RunningStage.VALIDATING)

            # Extract signals
            input_signals = batch['decode']
            predicted_signals = pred

            all_inputs.append(input_signals.cpu().numpy())
            all_predictions.append(predicted_signals.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                processed = min((batch_idx + 1) * dataloader.batch_size, total_samples)
                print(f"  Processed {processed} / {total_samples} samples...")

    # Concatenate batches → shape: (num_windows, channels, window_size)
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    print(f"✓ Collected {all_inputs.shape[0]} windows")

    # Create stitched sequences for visualization
    stitched_input = stitch_windows(all_inputs, stride, window_size)
    stitched_predictions = stitch_windows(all_predictions, stride, window_size)
    print(f"✓ Stitched into continuous sequences: {stitched_input.shape}")

    # Stats for denormalization are provided by the datamodule.
    stats = getattr(datamodule, "decode_stats_by_name", None) if datamodule is not None else None

    # Return both raw windows (for downstream tasks) and stitched sequences (for visualization)
    return all_inputs, all_predictions, stitched_input, stitched_predictions, stats


def stitch_windows(windows, stride, window_size):
    """
    Stitch overlapping windows into continuous sequences using averaging.

    Uses EnCodec's approach: average overlapping regions to smooth discontinuities.

    Args:
        windows: Array of windows (num_windows, channels, window_size)
        stride: Stride used during windowing
        window_size: Size of each window

    Returns:
        stitched: Continuous sequence (channels, total_length)

    Note:
        Assumes windows are consecutive in time (requires shuffle=False in dataloader).

        Strategy:
        - Non-overlapping portion of each window: kept as-is
        - Overlapping portion: averaged between adjacent windows

        Example (stride=4, window_size=8):
          Window 0: [a, b, c, d, e, f, g, h]
          Window 1:             [e', f', g', h', i, j, k, l]

          Result: [a, b, c, d, avg(e,e'), avg(f,f'), avg(g,g'), avg(h,h'), i, j, k, l]
    """
    num_windows, num_channels, _ = windows.shape

    if stride >= window_size:
        # Non-overlapping: simple concatenation
        return np.concatenate([windows[i] for i in range(num_windows)], axis=1)

    # Calculate overlap size
    overlap = window_size - stride

    # Calculate total length: (num_windows - 1) * stride + window_size
    total_length = (num_windows - 1) * stride + window_size

    # Initialize output array
    stitched = np.zeros((num_channels, total_length))

    # Stitch windows with averaging in overlapping regions
    for i in range(num_windows):
        start_pos = i * stride

        if i == 0:
            # First window: keep everything
            stitched[:, start_pos:start_pos + window_size] = windows[i]
        else:
            # Subsequent windows: average overlap, keep non-overlap
            # Overlap region: [start_pos : start_pos + overlap]
            # Non-overlap region: [start_pos + overlap : start_pos + window_size]

            # Average the overlapping portion
            stitched[:, start_pos:start_pos + overlap] = (
                stitched[:, start_pos:start_pos + overlap] + windows[i, :, :overlap]
            ) / 2.0

            # Keep the non-overlapping portion
            stitched[:, start_pos + overlap:start_pos + window_size] = windows[i, :, overlap:]

    return stitched


def compute_metrics(input_windows, output_windows, stitched_input, stitched_output, stats, signal_names, model, model_name, quantizer_type, dataloader, config, output_dir):
    """ Compute all metrics."""

    all_metrics = {}

    # Decoder loss analysis (uses stitched sequences)
    print("\n" + "="*60)
    print("DECODER LOSS ANALYSIS")
    print("="*60)
    all_metrics["decoder_loss_stats"] = decoder_loss_analysis(stitched_input, stitched_output, stats, signal_names)

    # Compression statistics
    print("\n" + "="*60)
    print("COMPRESSION ANALYSIS")
    print("="*60)
    all_metrics["compression_stats"] = collect_compression_stats(model, model_name, quantizer_type, dataloader, config)

    # Downstream task evaluation (uses raw windows)
    print("\n" + "="*60)
    print("DOWNSTREAM TASK UTILITY")
    print("="*60)
    all_metrics["downstream_task_stats"] = evaluate_downstream_task(input_windows, output_windows, config, dataloader, output_dir)

    return all_metrics


def decoder_loss_analysis(input_np, pred_np, stats, signal_names):
    """
    Calculate reconstruction metrics on stitched continuous sequences.
    """
    n_channels, total_timesteps = input_np.shape

    # Normalized MSE (standardized space)
    # Formula: MSE = (1/T) × Σ_d Σ_t (x - x̂)²
    # Sum over channels, mean over time
    squared_errors = (input_np - pred_np) ** 2  # (n_channels, total_timesteps)
    mse_per_channel = squared_errors.mean(axis=1)  # Mean over time per channel
    normalized_mse_summed = mse_per_channel.sum()  # Sum over channels

    # Per-channel metrics
    metrics = {
        'normalized_mse_summed': normalized_mse_summed,
        'per_channel': {}
    }

    for i, name in enumerate(signal_names[:n_channels]):
        # Extract channel data
        input_channel = input_np[i, :]  # (total_timesteps,)
        pred_channel = pred_np[i, :]  # (total_timesteps,)

        # Variance analysis (normalized space)
        signal_variance = np.var(input_channel)
        signal_std = np.std(input_channel)

        # Predictability: Autocorrelation (lag-1)
        # Measures temporal structure: high = predictable, low = random
        signal_flat = input_channel.flatten()
        if len(signal_flat) > 1:
            autocorr = np.corrcoef(signal_flat[:-1], signal_flat[1:])[0, 1]
        else:
            autocorr = 0.0

        # Normalized metrics
        mse_norm = np.mean((input_channel - pred_channel) ** 2)
        rmse_norm = np.sqrt(mse_norm)
        mae_norm = np.mean(np.abs(input_channel - pred_channel))

        # Correlation
        correlation = np.corrcoef(input_channel.flatten(), pred_channel.flatten())[0, 1]

        # Denormalized metrics (if stats available)
        if stats and name in stats:
            mean = float(stats[name]['mean'])
            std = float(stats[name]['std'])

            # Denormalize
            input_denorm = input_channel * std + mean
            pred_denorm = pred_channel * std + mean

            # Original scale metrics
            mse_orig = np.mean((input_denorm - pred_denorm) ** 2)
            rmse_orig = np.sqrt(mse_orig)

            # MAPE - Mean Absolute Percentage Error - Percentage error of reconstructed to original, averaged per channel
            mape = np.mean(np.abs((input_denorm - pred_denorm) / (np.abs(input_denorm) + 1e-8))) * 100

            metrics['per_channel'][name] = {
                'signal_variance': signal_variance,
                'signal_std': signal_std,
                'autocorrelation': autocorr,
                'mse_normalized': mse_norm,
                'rmse_normalized': rmse_norm,
                'mae_normalized': mae_norm,
                'correlation': correlation,
                'mse_original': mse_orig,
                'rmse_original': rmse_orig,
                'rmse_pct_std': (rmse_orig / std) * 100,
                'mape_pct': mape
            }
        else:
            metrics['per_channel'][name] = {
                'signal_variance': signal_variance,
                'signal_std': signal_std,
                'autocorrelation': autocorr,
                'mse_normalized': mse_norm,
                'rmse_normalized': rmse_norm,
                'mae_normalized': mae_norm,
                'correlation': correlation
            }

    return metrics


def collect_compression_stats(model, model_name, quantizer_type, dataloader, config):
    """
    Collect compression statistics per window.r
    """

    window_size = config.dataset.arguments.window_size
    in_channels = len(config.dataset.arguments.encode_signals)

    if model_name == "baseline":
        # Baseline model: continuous embedding compression (no quantization)
        embedding_dim = config.model.arguments.embedding_dim

        # Uncompressed: 32-bit floats × channels × window_size
        uncompressed_bits = 32 * in_channels * window_size

        # Compressed: 32-bit floats × embedding_dim × window_size
        compressed_bits = 32 * embedding_dim * window_size
        compression_ratio = uncompressed_bits / compressed_bits

        return {
            'quantizer_type': 'baseline',
            'compression_ratio': compression_ratio,
            'embedding_dim': embedding_dim,
            'uncompressed_bits_per_window': uncompressed_bits,
            'compressed_bits_per_window': compressed_bits
        }

    # Tokenizer models: extract quantizer
    quantizer = model.quantizer_module
    codebook_size = quantizer.codebook_size

    # Helper function to extract and prepare signals
    def prepare_signals(batch):
        return batch['encode']

    # Collect statistics
    total_windows = 0
    total_codes = 0
    routing_distribution = None

    with torch.no_grad():
        for batch in dataloader:
            input_signals = prepare_signals(batch)
            batch_size = input_signals.shape[0]
            total_windows += batch_size

            if quantizer_type == "rvq":
                # ResidualVectorQuantizer: Multiple quantizers per channel
                # embedded shape: (batch, num_channels, emb_dim) e.g., (batch, 9, 72)
                # RVQ treats this as: (batch, sequence=num_channels, dim=emb_dim)
                # Output indices: (batch, num_channels, num_quantizers)
                # Each channel gets num_quantizers codes (one from each RVQ layer)
                embedded = model.encode(input_signals)
                
                num_channels = embedded.shape[1]
                num_quantizers = quantizer.num_quantizers
                codes_per_window = num_channels * num_quantizers 
                total_codes += batch_size * codes_per_window

            elif quantizer_type == "signal_adaptive_vq":
                # Signal Adaptive VQ: Partitioned with 1 quantizer per channel
                # embedded shape: (batch, num_channels, emb_dim) e.g., (batch, 9, 72)
                # Output indices: (batch, num_channels) - one index per channel
                # Each channel gets 1 code from its partition-specific codebook
                
                # Prepare signal_types tensor
                signal_types = model._signal_types_base.unsqueeze(0).expand(batch_size, -1)

                # Encode with signal types
                embedded = model.encode(input_signals, signal_types)
                
                num_channels = embedded.shape[1]
                num_quantizers = quantizer.num_quantizers  
                codes_per_window = num_channels * num_quantizers
                total_codes += batch_size * codes_per_window

    # Calculate metrics
    avg_codes_per_window = total_codes / total_windows
    bits_per_code = np.log2(codebook_size)
    avg_bits_per_window = avg_codes_per_window * bits_per_code
    avg_bits_per_timestep = avg_bits_per_window / window_size

    # Uncompressed: 32-bit floats × channels × window_size
    uncompressed_bits = 32 * in_channels * window_size
    compression_ratio = uncompressed_bits / avg_bits_per_window

    result = {
        'total_windows': total_windows,
        'avg_codes_per_window': avg_codes_per_window,
        'bits_per_code': bits_per_code,
        'avg_bits_per_window': avg_bits_per_window,
        'avg_bits_per_timestep': avg_bits_per_timestep,
        'compression_ratio': compression_ratio,
        'codebook_size': codebook_size
    }

    # Add routing info if available
    if routing_distribution:
        result['routing_distribution'] = routing_distribution
        result['routing_percentages'] = {
            k: 100.0 * v / total_windows for k, v in routing_distribution.items()
        }

    return result


def load_downstream_targets(config, dataloader):
    """
    Load downstream task targets based on dataset type.

    Hardcoded task mappings:
    - emob dataset → Binary Classification (anomaly detection)
    - smart home dataset → Regression (appliance energy prediction)

    Args:
        config: Dataset configuration
        dataloader: Test dataloader to align targets with windows

    Returns:
        targets: numpy array of target values
        task_type: 'classification' or 'regression' (hardcoded per dataset)
        task_name: descriptive name of the task
    """
    dataset_name = config.dataset.name
    split_ratios = config.dataset.arguments.split_ratios
    train_ratio, val_ratio, test_ratio = split_ratios

    # ============================================================================
    # EMOB DATASET - CLASSIFICATION TASK (Anomaly Detection)
    # ============================================================================
    if dataset_name == 'emob' or 'emob' in str(config.dataset.arguments.get('file_path', '')).lower():
        TASK_TYPE = 'classification'  # Hardcoded
        TASK_NAME = 'Battery Anomaly Detection'

        # Label file can be specified in config or use default location
        LABEL_FILE = config.dataset.arguments.get(
            'label_file',
            "datasets/emob_cycle_labels.csv"
        )

        print(f"Dataset: emob | Task: {TASK_NAME} ({TASK_TYPE})")
        print(f"Loading labels from: {LABEL_FILE}")

        # Load anomaly labels
        labels_df = pd.read_csv(LABEL_FILE)
        total_windows = len(labels_df)

        # Calculate test set start index
        val_end = int(total_windows * (train_ratio + val_ratio))

        # Extract test set labels (binary: 0=normal, 1=anomaly)
        test_labels = labels_df.iloc[val_end:]['anomaly_label'].values

        # Align with dataloader size
        num_windows = len(dataloader.dataset)
        targets = test_labels[:num_windows]

        return targets, TASK_TYPE, TASK_NAME

    # ============================================================================
    # SMART HOME DATASET - REGRESSION TASK (Energy Prediction)
    # ============================================================================
    elif dataset_name == 'timeseries_text' or 'energydata' in str(config.dataset.arguments.get('file_path', '')).lower():
        TASK_TYPE = 'regression'  # Hardcoded
        TASK_NAME = 'Appliance Energy Sequence Prediction'
        TARGET_COLUMN = 'Appliances'

        if 'file_path' not in config.dataset.arguments:
            raise ValueError("Smart home dataset requires 'file_path' in config")

        file_path = config.dataset.arguments.file_path

        print(f"Dataset: smart home | Task: {TASK_NAME} ({TASK_TYPE})")
        print(f"Loading target column '{TARGET_COLUMN}' from: {file_path}")

        df = pd.read_csv(file_path)
        appliances = df[TARGET_COLUMN].values

        window_size = config.dataset.arguments.window_size
        stride = config.dataset.arguments.stride

        num_timesteps = len(df)
        num_windows = (num_timesteps - window_size) // stride + 1

        val_end = int(num_windows * (train_ratio + val_ratio))
 
        test_window_targets = []
        for i in range(val_end, num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            window_appliances = appliances[start_idx:end_idx]
            test_window_targets.append(window_appliances)  # Full sequence

        targets = np.array(test_window_targets)  # (num_test_windows, window_size)

        return targets, TASK_TYPE, TASK_NAME

    # ============================================================================
    # UNKNOWN DATASET
    # ============================================================================
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}\n"
            f"Supported datasets:\n"
            f"  - 'emob' → Binary classification (anomaly detection)\n"
            f"  - 'timeseries_text' with 'energydata' file → Regression (energy prediction)"
        )


def train_downstream_predictor(features, targets, task_type):
    """
    Train a prediction model for downstream task evaluation using sklearn.

    IMPLEMENTATION:
    - Classification: RandomForestClassifier
    - Regression: RandomForestRegressor

    Args:
        features: Input features (num_windows, num_channels, window_size)
        targets: Target labels/values (num_windows,)
        task_type: 'classification' or 'regression'

    Returns:
        model: Trained predictor model
        train_metrics: Training metrics dict
        test_metrics: Test metrics dict
        predictions: Dict with 'y_true' and 'y_pred' for test set (for visualization)
    """
    # Flatten features for sklearn: (num_windows, num_channels * window_size)
    num_windows = features.shape[0]
    X_flat = features.reshape(num_windows, -1)

    # Train/test split (only stratify if classification and multiple classes exist)
    stratify = None
    if task_type == 'classification' and len(np.unique(targets)) > 1:
        stratify = targets

    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, targets, test_size=0.2, random_state=42, stratify=stratify
    )

    if task_type == 'classification':
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            class_weight='balanced',  # Handles class imbalance (e.g., rare anomalies)
            n_jobs=-1                 # Use all CPU cores
        )
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='binary', zero_division=0
        )

        try:
            auc = roc_auc_score(y_test, y_test_proba)
        except ValueError:
            auc = 0.0

        train_metrics = {'accuracy': train_acc}
        test_metrics = {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    else:  # regression
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_r2 = r2_score(y_train, y_train_pred, multioutput='uniform_average')
        test_r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)

        # Per-timestep metrics: how well does each position predict?
        per_timestep_r2 = r2_score(y_test, y_test_pred, multioutput='raw_values')

        # Also compute a "temporal R²" averaged across timesteps
        # This tells us: how well does the predicted *shape* match?
        # Normalized per-timestep correlation
        timestep_corrs = []
        for t in range(y_test.shape[1]):
            corr = np.corrcoef(y_test[:, t], y_test_pred[:, t])[0, 1]
            timestep_corrs.append(corr)

        train_metrics = {
            'mse': mean_squared_error(y_train, y_train_pred),
            'r2': train_r2
        }
        test_metrics = {
            'mse': test_mse,
            'mae': test_mae,
            'r2': test_r2,
            'rmse': test_rmse,
            'per_timestep_r2_mean': per_timestep_r2.mean(),
            'per_timestep_r2_min': per_timestep_r2.min(),
            'per_timestep_r2_max': per_timestep_r2.max(),
            'per_timestep_corr_mean': np.mean(timestep_corrs),
            'per_timestep_corr_min': np.min(timestep_corrs),
            'target_sequence_length': y_test.shape[1],
        }

        print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Test MAE: {test_mae:.4f}")
        print(f"  Per-timestep R² — Mean: {per_timestep_r2.mean():.4f}, Min: {per_timestep_r2.min():.4f}, Max: {per_timestep_r2.max():.4f}")
        print(f"  Per-timestep Corr — Mean: {np.mean(timestep_corrs):.4f}, Min: {np.min(timestep_corrs):.4f}")

    predictions = {
        'y_true': y_test,
        'y_pred': y_test_pred
    }

    return model, train_metrics, test_metrics, predictions


def evaluate_downstream_task(input_windows, output_windows, config, dataloader, output_dir):
    """
    Evaluate downstream task utility by comparing model performance
    on original vs. reconstructed signals.

    Args:
        input_windows: Ground truth raw windows (num_windows, num_channels, window_size)
        output_windows: Reconstructed raw windows (num_windows, num_channels, window_size)
        config: Dataset configuration
        dataloader: Test dataloader for alignment
        output_dir: Directory for saving outputs

    Returns:
        metrics: Dict with downstream task performance comparison
    """
    try:
        # Load targets for downstream task
        targets, task_type, task_name = load_downstream_targets(config, dataloader)

        print(f"\nDownstream Task: {task_name}")
        print(f"Task Type: {task_type}")
        print(f"Number of samples: {len(targets)}")

        # Align with targets (handle potential size mismatch)
        min_samples = min(len(input_windows), len(targets))
        input_windows = input_windows[:min_samples]
        output_windows = output_windows[:min_samples]
        targets = targets[:min_samples]

        print(f"Aligned windows shape: {input_windows.shape}")
        print(f"Targets shape: {targets.shape}")

        # Train predictor on original signals
        print("\n[1/2] Training predictor on ORIGINAL signals...")
        original_model, original_train_metrics, original_test_metrics, original_predictions = train_downstream_predictor(
            input_windows, targets, task_type
        )

        # Train predictor on reconstructed signals
        print("\n[2/2] Training predictor on RECONSTRUCTED signals...")
        reconstructed_model, reconstructed_train_metrics, reconstructed_test_metrics, reconstructed_predictions = train_downstream_predictor(
            output_windows, targets, task_type
        )

        # Compute utility retention
        if task_type == 'classification':
            # For classification, compare test accuracy
            original_score = original_test_metrics['accuracy']
            reconstructed_score = reconstructed_test_metrics['accuracy']
            metric_name = 'accuracy'
        else:
            # For regression, compare R2 score (higher is better)
            original_score = original_test_metrics['r2']
            reconstructed_score = reconstructed_test_metrics['r2']
            metric_name = 'r2'

        utility_retention = (reconstructed_score / original_score) * 100 if original_score > 0 else 0.0

        results = {
            'task_name': task_name,
            'task_type': task_type,
            'num_samples': min_samples,
            'original_performance': original_test_metrics,
            'reconstructed_performance': reconstructed_test_metrics,
            'utility_retention_pct': utility_retention,
            'primary_metric': metric_name
        }

        # Print summary
        print("\n" + "="*60)
        print(f"DOWNSTREAM TASK RESULTS: {task_name}")
        print("="*60)
        print(f"\nOriginal Signals {metric_name.upper()}: {original_score:.4f}")
        print(f"Reconstructed Signals {metric_name.upper()}: {reconstructed_score:.4f}")
        print(f"Utility Retention: {utility_retention:.2f}%")

        return results

    except Exception as e:
        print(f"\n⚠ Downstream task evaluation failed: {e}")
        print("Skipping downstream task metrics...")
        return {
            'error': str(e),
            'task_available': False
        }




def visualize_long_sequences(input_np, pred_np, output_path: str, signal_names, sequence_length=1024, num_sequences=5):
    """
    Visualize multiple long sequences from stitched continuous data.

    Args:
        input_np: Ground truth stitched sequence (channels, total_timesteps)
        pred_np: Predicted stitched sequence (channels, total_timesteps)
        output_path: Path to save visualization
        signal_names: List of signal names
        sequence_length: Length of each sequence to plot (default: 1024 timesteps)
        num_sequences: Number of different sequences to plot (default: 5)
    """
    print("\n" + "="*60)
    print("GENERATING LONG SEQUENCE VISUALIZATIONS")
    print("="*60)

    n_channels, total_timesteps = input_np.shape

    if signal_names is None:
        signal_names = [f"Signal_{i}" for i in range(n_channels)]

    # Find starting indices for multiple sequences across the dataset
    max_start = total_timesteps - sequence_length
    if max_start <= 0:
        print(f"⚠ Total length ({total_timesteps}) < sequence_length ({sequence_length}), using full sequence")
        sequence_length = total_timesteps
        start_indices = [0]
        num_sequences = 1
    else:
        # Distribute sequences evenly: beginning, 1/4, middle, 3/4, end
        start_indices = np.linspace(0, max_start, num_sequences, dtype=int)

    print(f"\nPlotting {num_sequences} sequences of {sequence_length} timesteps each")
    print(f"Total continuous sequence length: {total_timesteps} timesteps")
    print(f"Starting at timestep indices: {start_indices.tolist()}")

    # Create figure with subplots: num_sequences rows × n_channels columns
    fig, axes = plt.subplots(num_sequences, n_channels, figsize=(8*n_channels, 4*num_sequences))

    # Handle edge cases for subplot indexing
    if num_sequences == 1 and n_channels == 1:
        axes = np.array([[axes]])
    elif num_sequences == 1:
        axes = axes.reshape(1, -1)
    elif n_channels == 1:
        axes = axes.reshape(-1, 1)

    for seq_idx, start_idx in enumerate(start_indices):
        # Extract sequence slice from stitched data
        end_idx = start_idx + sequence_length
        long_input = input_np[:, start_idx:end_idx]  # (channels, sequence_length)
        long_pred = pred_np[:, start_idx:end_idx]  # (channels, sequence_length)

        for channel_idx in range(n_channels):
            ax = axes[seq_idx, channel_idx]

            input_seq = long_input[channel_idx, :]
            pred_seq = long_pred[channel_idx, :]
            timesteps = np.arange(len(input_seq))

            # Plot ground truth and predictions
            ax.plot(timesteps, input_seq, label='Ground Truth', linewidth=1.5, alpha=0.7, color='#2E86AB')
            ax.plot(timesteps, pred_seq, label='Prediction', linewidth=1.5, alpha=0.7, linestyle='--', color='#A23B72')

            # Compute metrics for this sequence
            mse = np.mean((input_seq - pred_seq) ** 2)
            corr = np.corrcoef(input_seq, pred_seq)[0, 1]
            mae = np.mean(np.abs(input_seq - pred_seq))

            # Title with sequence info
            sequence_label = ["Beginning", "1/4", "Middle", "3/4", "End"][seq_idx] if num_sequences == 5 else f"Seq {seq_idx+1}"
            ax.set_title(f'{signal_names[channel_idx]} - {sequence_label} (Timesteps {start_idx}-{end_idx})\n'
                        f'Corr: {corr:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}', fontsize=10)

            if seq_idx == num_sequences - 1:  # Only label x-axis on bottom row
                ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('Normalized Value', fontsize=9)

            if seq_idx == 0 and channel_idx == 0:  # Only show legend once
                ax.legend(fontsize=9, loc='best')

            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Long sequence visualization saved: {output_path}")

    # Print summary statistics across all sequences
    print(f"\nCorrelation Summary Across {num_sequences} Sequences:")
    for channel_idx, signal_name in enumerate(signal_names):
        correlations = []
        for start_idx in start_indices:
            # Extract sequence from stitched data
            end_idx = start_idx + sequence_length
            input_seq = input_np[channel_idx, start_idx:end_idx]
            pred_seq = pred_np[channel_idx, start_idx:end_idx]
            corr = np.corrcoef(input_seq, pred_seq)[0, 1]
            correlations.append(corr)

        print(f"  {signal_name}: Mean={np.mean(correlations):.4f}, Std={np.std(correlations):.4f}, "
              f"Min={np.min(correlations):.4f}, Max={np.max(correlations):.4f}")


def visualize_ground_truth_only(input_np, output_path, signal_names, sequence_length=1024, num_sequences=5):
    """Visualize ground truth only without reconstruction."""
    n_channels, total_timesteps = input_np.shape

    if signal_names is None:
        signal_names = [f"Signal_{i}" for i in range(n_channels)]

    # Find starting indices for multiple sequences across the dataset
    max_start = total_timesteps - sequence_length
    if max_start <= 0:
        print(f"⚠ Total length ({total_timesteps}) < sequence_length ({sequence_length}), using full sequence")
        sequence_length = total_timesteps
        start_indices = [0]
        num_sequences = 1
    else:
        # Distribute sequences evenly: beginning, 1/4, middle, 3/4, end
        start_indices = np.linspace(0, max_start, num_sequences, dtype=int)

    # Create figure with subplots: num_sequences rows × n_channels columns
    fig, axes = plt.subplots(num_sequences, n_channels, figsize=(8*n_channels, 4*num_sequences))

    # Handle edge cases for subplot indexing
    if num_sequences == 1 and n_channels == 1:
        axes = np.array([[axes]])
    elif num_sequences == 1:
        axes = axes.reshape(1, -1)
    elif n_channels == 1:
        axes = axes.reshape(-1, 1)

    for seq_idx, start_idx in enumerate(start_indices):
        # Extract sequence slice from stitched data
        end_idx = start_idx + sequence_length
        long_input = input_np[:, start_idx:end_idx]  # (channels, sequence_length)

        for channel_idx in range(n_channels):
            ax = axes[seq_idx, channel_idx]

            input_seq = long_input[channel_idx, :]
            timesteps = np.arange(len(input_seq))

            # Plot ground truth and predictions
            ax.plot(timesteps, input_seq, label='Original Signal', linewidth=1.5, alpha=0.7, color='#2E86AB')

            # Title with sequence info
            sequence_label = ["Beginning", "1/4", "Middle", "3/4", "End"][seq_idx] if num_sequences == 5 else f"Seq {seq_idx+1}"
            ax.set_title(f'{signal_names[channel_idx]} - {sequence_label} (Timesteps {start_idx}-{end_idx})', fontsize=10)

            if seq_idx == num_sequences - 1:  # Only label x-axis on bottom row
                ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('Normalized Value', fontsize=9)

            if seq_idx == 0 and channel_idx == 0:  # Only show legend once
                ax.legend(fontsize=9, loc='best')

            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

def save_to_file(metrics_path, config_path, checkpoint_path, metadata, metrics):
    """ Saves the evaluation results to a file. """
    print("\n" + "="*60)
    print("WRITING TO FILE")
    print("="*60)

    decoder_loss_stats = metrics["decoder_loss_stats"]
    compression_stats = metrics["compression_stats"]
    downstream_stats = metrics.get("downstream_task_stats", None)

    with open(metrics_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Metadata: {metadata}\n\n")

        # Decoder loss analysis
        f.write("="*60 + "\n")
        f.write("DECODER LOSS ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Normalized MSE (summed): {decoder_loss_stats['normalized_mse_summed']:.6f}\n")

        f.write("\nPer-Channel Metrics:\n")
        f.write("-" * 60 + "\n")
        for name, channel_metrics in decoder_loss_stats['per_channel'].items():
            f.write(f"\n{name}:\n")
            for metric_name, value in channel_metrics.items():
                f.write(f"  {metric_name}: {value:.6f}\n")

        # Compression metrics
        f.write("\n" + "="*60 + "\n")
        f.write("COMPRESSION ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model Type: {metadata['model_name']}\n")

        if metadata['model_name'] == 'baseline':
            f.write("\nBaseline Compression (continuous embedding):\n")
            f.write(f"  Embedding dimension: {compression_stats['embedding_dim']}\n")
            f.write(f"  Uncompressed bits per window: {compression_stats['uncompressed_bits_per_window']:.0f}\n")
            f.write(f"  Compressed bits per window: {compression_stats['compressed_bits_per_window']:.0f}\n")
            f.write(f"  Compression ratio: {compression_stats['compression_ratio']:.1f}x\n")
        else:
            if 'routing_distribution' in compression_stats:
                f.write("\nRouting Distribution:\n")
                for route, pct in compression_stats['routing_percentages'].items():
                    count = compression_stats['routing_distribution'][route]
                    f.write(f"  {route.capitalize()}: {count:,} windows ({pct:.1f}%)\n")

            f.write("\nCompression Metrics:\n")
            f.write(f"  Average codes per window: {compression_stats['avg_codes_per_window']:.2f}\n")
            f.write(f"  Bits per code: {compression_stats['bits_per_code']:.1f}\n")
            f.write(f"  Average bits per window: {compression_stats['avg_bits_per_window']:.1f}\n")
            f.write(f"  Average bits per timestep: {compression_stats['avg_bits_per_timestep']:.3f}\n")
            f.write(f"  Compression ratio: {compression_stats['compression_ratio']:.1f}x\n")

        # Downstream task metrics
        if downstream_stats and 'task_available' not in downstream_stats:
            f.write("\n" + "="*60 + "\n")
            f.write("DOWNSTREAM TASK UTILITY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Task: {downstream_stats['task_name']}\n")
            f.write(f"Task Type: {downstream_stats['task_type']}\n")
            f.write(f"Samples: {downstream_stats['num_samples']}\n\n")

            f.write("Original Signal Performance:\n")
            for metric_name, value in downstream_stats['original_performance'].items():
                f.write(f"  {metric_name}: {value:.6f}\n")

            f.write("\nReconstructed Signal Performance:\n")
            for metric_name, value in downstream_stats['reconstructed_performance'].items():
                f.write(f"  {metric_name}: {value:.6f}\n")

            f.write(f"\nUtility Retention: {downstream_stats['utility_retention_pct']:.2f}%\n")

    print(f"\n✓ Metrics saved: {metrics_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained neural compression model')
    parser.add_argument('config', type=str, help='Path to training config YAML')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--num-vis-samples', type=int, default=3,
                        help='Number of samples to visualize (default: 3)')

    args = parser.parse_args()

    # Resolve paths 
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    # Load config and setup metadata
    config = OmegaConf.load(config_path)

    output_dir = checkpoint_path.parent
    dataset = config.dataset.name
    model_name = config.model.name
    encoder_type = config.model.arguments.encoder.type
    quantizer_type = config.model.arguments.get('quantizer', {}).get('type', None)

    metadata = {
        "output_dir": output_dir,
        "dataset": dataset,
        "model_name": model_name,
        "encoder_type": encoder_type,
        "quantizer_type": quantizer_type,
    } 

    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"\nConfig:      {config_path}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Output dir:  {output_dir}")

    # Load model and data
    runner, datamodule = load_checkpoint_and_data(config, str(checkpoint_path))
    model = runner.training_module.model
    dataloader = datamodule.test_dataloader()

    # Get signal names, stride, and window_size from config
    signal_names = config.dataset.arguments.encode_signals
    stride = config.dataset.arguments.stride
    window_size = config.dataset.arguments.window_size

    # Evaluate full dataset
    input_windows, output_windows, stitched_input, stitched_output, stats = evaluate_full_dataset(
        runner,
        dataloader,
        stride,
        window_size,
        datamodule=datamodule,
    )

    # Compute metrics (uses both raw windows for downstream tasks and stitched for analysis)
    metrics = compute_metrics(input_windows, output_windows, stitched_input, stitched_output, stats, signal_names, model, model_name, quantizer_type, dataloader, config, output_dir)

    # Long sequence visualization (uses stitched sequences)
    long_vis_path = output_dir / "reconstruction_sequence.png"
    visualize_long_sequences(stitched_input, stitched_output, str(long_vis_path), signal_names, sequence_length=1024, num_sequences=5)

    # Ground truth only visualization (uses stitched sequences)
    gt_vis_path = output_dir / "ground_truth_sequence.png"
    visualize_ground_truth_only(stitched_input, str(gt_vis_path), signal_names, sequence_length=1024, num_sequences=5)

    # Save metrics to file
    metrics_path = output_dir / "metrics_test.txt"
    save_to_file(metrics_path, config_path, checkpoint_path, metadata, metrics)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
