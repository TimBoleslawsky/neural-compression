import argparse
import os
import sys
from pathlib import Path
import torch as th
import numpy as np
import random
import lightning.pytorch as pl
import torch
from dotenv import load_dotenv

from framework.utils import (
    model_factory,
    dataset_factory,
    training_module_factory,
    optimizer_factory,
    scheduler_factory,
    callback_factory,
    runner_factory,
    TrainingArgs
)

# Prepare sys.path
sys.path = [os.getcwd()] + sys.path
sys.path = ["framework"] + sys.path

def main():
    config_args = _parse_console_args()

    # Local single-node training only (Ray support removed)
    train_func(config_args)


def train_func(train_loop_config):
    """Training function.

    Args:
        train_loop_config: TrainingArgs configuration
    """
    # Part 1: Environment Setup - Initialize training environment, seeds, and logging
    config, logger = _setup_training_environment(train_loop_config)

    # Part 2: Component Assembly - Create datasets, models, runners, and trainer
    signal_data_module, runner, pl_trainer = _assemble_training_components(config, logger)

    # Part 3: Training Execution - Load checkpoints and run training/testing
    if config.checkpoints["trainer"] is not None:
        if (config.mode == "finetune") and (config.checkpoints.model is not None):
            raise ValueError(
                "Finetuning mode was set but both model and training state checkpoints are given.",
                "The training state checkpoint would overwrite the weights of the model to-be-fine-tuned,",
                "therefore only either the model or the training state checkpoint can be given!",
            )

    if config.mode in ["train", "finetune"]:
        pl_trainer.fit(
            model=runner,
            datamodule=signal_data_module,
            ckpt_path=config.checkpoints.trainer,
        )
    else:
        pl_trainer.test(
            model=runner,
            datamodule=signal_data_module,
            ckpt_path=config.checkpoints.trainer,
        )

def _parse_console_args():
    parser = argparse.ArgumentParser(description="Train configuration")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="the (relative) path to the yaml config",
    )
    args = parser.parse_args()

    # Treat `--config` as a path relative to the current working directory.
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    else:
        config_path = config_path.resolve()

    config_args = TrainingArgs(config_path)

    return config_args


def _setup_training_environment(train_loop_config):
    """Part 1: Setup training environment, seeds, and logging.

    Args:
        train_loop_config: TrainingArgs configuration

    Returns:
        tuple: (config, logger)
    """
    load_dotenv()

    # Set seeds for reproducibility
    seed = train_loop_config.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

    if train_loop_config is None:
        raise ValueError("train_loop_config cannot be None")

    config = train_loop_config

    # MLflow support removed
    logger = None

    return config, logger


def _assemble_training_components(config, logger):
    """Part 2: Assemble all training components (data, model, runner, trainer).

    Args:
        config: TrainingArgs configuration
        logger: Logger (unused; kept for Trainer API)

    Returns:
        tuple: (signal_data_module, runner, pl_trainer)
    """
    # Optimize for Tensor Cores (H100) - hardware optimization, not changing paper specs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # Use Tensor Cores for matmul

    # Create dataset/dataloader module
    signal_data_module = dataset_factory(
        **config.dataset, dataloader_arguments=config.dataloader
    )

    # Create and setup model (pass dataset config for shared parameters like window_size)
    model: torch.nn.Module = model_factory(**config.model, dataset_config=config.dataset)

    # Create task runner (combines model + training module + optimizer + scheduler)
    runner_parameters = {
        "training_module": training_module_factory(
            model,
            **config.training_module,
            model_config=config.model,
            full_config=config
        ),
        "optimizer": optimizer_factory(**config.optimizer),
        "scheduler": (
            scheduler_factory(**config.scheduler)
            if config.scheduler is not None
            else None
        ),
    }
    if config.runner.parameters is not None:
        runner_parameters.update(config.runner.parameters)
    runner = runner_factory(name=config.runner.name, parameters=runner_parameters)

    # Setup callbacks
    callbacks = callback_factory(config.callbacks)

    # Local: Use config strategy or auto-detection
    default_strategy = config.trainer.strategy if hasattr(config.trainer, 'strategy') else "auto"
    trainer_config = {
        "accelerator": "auto",
        "devices": "auto",
        "strategy": default_strategy,
        "logger": logger,
        "callbacks": callbacks,
    }

    # Config overrides defaults
    config_trainer = dict(config.trainer)
    trainer_config.update(config_trainer)

    pl_trainer = pl.Trainer(**trainer_config)

    return signal_data_module, runner, pl_trainer


if __name__ == "__main__":
    main()
