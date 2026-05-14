from functools import partial
from typing import Union, Any

from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import LightningDataModule
from torch import nn, optim

# Import datasets/datamodules
from datasets import TimeSeriesTextLightningDataModule

# Import basic ML components, only two types
from framework.training import (
    TrainingModule,
    CompressionTrainingModule,
    EdgeCodecTrainingModule,
    UniEdgeCodecTrainingModule,
    TaskAwareEdgeCodecTrainingModule,
)

from framework.models import (
    BaselineCompressionModel,
    EdgeCodecModel,
    EdgeCodecDiscriminator,
    UniEdgeCodecModel,
)

from framework.models.encoders import (
    SimpleCNNEncoder,
    TCNEncoder,
    ResNetEncoder,
    EdgeCodecEncoder,
)

from framework.models.decoders import (
    GRUDecoder,
    EdgeCodecDecoder,
)

from framework.models.quantizers import (
    ResidualVectorQuantizer,
    SignalAdaptiveVQ
)

from framework.runner import (
    BaseRunner
)


def model_factory(name: str, arguments: dict, dataset_config: dict = None) -> nn.Module:
    """
    Create a model with encoder and optional quantizer from config.
    """
    # Convert OmegaConf DictConfig to plain dict to allow storing non-primitive objects
    arguments = dict(arguments)
    if dataset_config is not None:
        dataset_config = dict(dataset_config)

    encoder_config = arguments.pop("encoder", None)

    if not encoder_config:
        raise AttributeError("No encoder configuration supplied")

    encoder_type = encoder_config["type"]
    encoder_args = encoder_config["arguments"]

    # Create encoder
    if encoder_type == "simple_cnn":
        encoder_module = SimpleCNNEncoder(**encoder_args)
    elif encoder_type == "tcn":
        encoder_module = TCNEncoder(**encoder_args)
    elif encoder_type == "ResNet":
        encoder_module = ResNetEncoder(**encoder_args)
    elif encoder_type == "edgecodec_encoder":
        encoder_module = EdgeCodecEncoder(**encoder_args)
    else:
        raise AttributeError(f"Encoder type '{encoder_type}' is not supported")

    arguments["encoder_module"] = encoder_module

    if name == "baseline":
        # Extract decoder configuration
        decoder_config = arguments.pop("decoder", None)
        if not decoder_config:
            raise AttributeError("No decoder configuration supplied for baseline")

        decoder_type = decoder_config["type"]
        decoder_args = decoder_config["arguments"]

        if decoder_type == "gru":
            decoder_module = GRUDecoder(**decoder_args)
        else:
            raise AttributeError(
                f"Decoder type '{decoder_type}' is not supported for baseline. "
                f"Supported: 'gru'"
            )

        arguments["decoder_module"] = decoder_module
        return BaselineCompressionModel(**arguments)

    elif name == "edgecodec":
        quantizer_config = arguments.pop("quantizer", None)
        if not quantizer_config:
            raise AttributeError("No quantizer configuration supplied for EdgeCodec")

        quantizer_type = quantizer_config["type"]
        quantizer_args = quantizer_config["arguments"]

        if quantizer_type == "rvq":
            quantizer = ResidualVectorQuantizer(**quantizer_args)
        else:
            raise AttributeError(
                f"Quantizer type '{quantizer_type}' not supported for EdgeCodec. "
                f"EdgeCodec requires 'rvq' quantizer type."
            )

        arguments["quantizer_module"] = quantizer

        decoder_config = arguments.pop("decoder", None)
        if not decoder_config:
            raise AttributeError("No decoder configuration supplied for EdgeCodec")

        decoder_type = decoder_config["type"]
        decoder_args = decoder_config["arguments"]

        # Set window_size for linear-wise operations
        if not dataset_config:
            raise AttributeError("No dataset configuration supplied for EdgeCodec")

        window_size = dataset_config["arguments"]["window_size"]

        decoder_args["window_size"] = window_size

        if decoder_type == "edgecodec_decoder":
            decoder_module = EdgeCodecDecoder(**decoder_args)
        else:
            raise AttributeError(
                f"Decoder type '{decoder_type}' is not supported for EdgeCodec. "
                f"Supported: 'edgecodec_decoder'"
            )
        arguments["decoder_module"] = decoder_module

        return EdgeCodecModel(**arguments)

    elif name == "uni_edgecodec":
        # UniEdgeCodec: Signal-adaptive compression with partitioned VQ codebook
        quantizer_config = arguments.pop("quantizer", None)
        if not quantizer_config:
            raise AttributeError("No quantizer configuration supplied for UniEdgeCodec")

        quantizer_type = quantizer_config["type"]
        quantizer_args = quantizer_config["arguments"]

        if quantizer_type == "signal_adaptive_vq":
            # SignalAdaptiveVQ with partitioned codebook
            quantizer = SignalAdaptiveVQ(**quantizer_args)
        elif quantizer_type == "rvq":
            # Fallback: allow RVQ for baseline comparison
            quantizer = ResidualVectorQuantizer(**quantizer_args)
        else:
            raise AttributeError(
                f"Quantizer type '{quantizer_type}' not supported for UniEdgeCodec. "
                f"Supported: 'signal_adaptive_vq', 'rvq' (fallback)"
            )

        arguments["quantizer_module"] = quantizer

        # Decoder configuration
        decoder_config = arguments.pop("decoder", None)
        if not decoder_config:
            raise AttributeError("No decoder configuration supplied for UniEdgeCodec")

        decoder_type = decoder_config["type"]
        decoder_args = decoder_config["arguments"]

        # Set window_size for channel-wise linear operations
        if not dataset_config:
            raise AttributeError("No dataset configuration supplied for UniEdgeCodec")

        window_size = dataset_config["arguments"]["window_size"]
        decoder_args["window_size"] = window_size

        if decoder_type == "edgecodec_decoder":
            decoder_module = EdgeCodecDecoder(**decoder_args)
        else:
            raise AttributeError(
                f"Decoder type '{decoder_type}' is not supported for UniEdgeCodec. "
                f"Supported: 'edgecodec_decoder'"
            )
        arguments["decoder_module"] = decoder_module

        # Align signal type ids to channel order (encode_signals order).
        signal_type_map = arguments.pop("signal_type_map", None)
        if signal_type_map is None:
            raise AttributeError("UniEdgeCodec requires 'signal_type_map' in model arguments")

        signal_type_map = dict(signal_type_map)
        if not dataset_config:
            raise AttributeError("No dataset configuration supplied for UniEdgeCodec")

        encode_signals = list(dataset_config["arguments"]["encode_signals"])
        signal_type_ids = [int(signal_type_map[s]) for s in encode_signals]

        arguments["signal_type_ids"] = signal_type_ids

        return UniEdgeCodecModel(**arguments)

    else:
        raise AttributeError(f"Model type '{name}' is not supported")

def dataset_factory(
    name: str, arguments: dict, dataloader_arguments: dict
) -> LightningDataModule:
    # Make a mutable copy of arguments to allow modifications
    arguments = dict(arguments)

    # Remove signal_types from dataset arguments if present
    # Signal types are used by model/training module, not by dataloader
    # They're kept in config for documentation but shouldn't be passed to dataloader
    arguments.pop("signal_types", None)

    if name == "timeseries_text":
        return TimeSeriesTextLightningDataModule(
            **arguments,
            **dataloader_arguments,
        )
    else:
        raise AttributeError(f"Selected dataset type {name} is not supported, yet!")


def training_module_factory(
    model: nn.Module,
    name: str,
    parameters: dict,
    model_config: dict = None,
    full_config: dict = None,
) -> nn.Module:
    """
    Create a training module with the given model and parameters.

    Args:
        model: The neural network model to wrap
        name: The type of training module to create
        parameters: Configuration parameters for the module
        model_config: Full model configuration
        full_config: Complete configuration (for task-aware modules that need dataset config)

    Returns:
        Configured training module instance
    """
    # Make a mutable copy of parameters
    parameters = dict(parameters)

    module_mapping = {
        "base": TrainingModule,
        "compression": CompressionTrainingModule,
        "edgecodec": EdgeCodecTrainingModule,
        "uni_edgecodec": UniEdgeCodecTrainingModule,
        "task_aware_edgecodec": TaskAwareEdgeCodecTrainingModule,
    }

    if name not in module_mapping:
        raise AttributeError(
            f"Training module type '{name}' is not supported. "
            f"Supported types: {', '.join(module_mapping.keys())}"
        )

    # Task-aware module needs full config for target loading
    if name == "task_aware_edgecodec":
        return module_mapping[name](model=model, config=full_config, **parameters)

    # EdgeCodec with optional discriminator
    elif name == "edgecodec":
        if model_config is not None:
            model_config = dict(model_config)

        if model_config["arguments"]["use_discriminator"]:
            discriminator = EdgeCodecDiscriminator(input_channels=model_config["arguments"]["encoder"]["arguments"]["in_channels"])
            return module_mapping[name](model=model, discriminator=discriminator, **parameters)
        else:
            return module_mapping[name](model=model, **parameters)

    elif name == "uni_edgecodec":
        # Align signal type ids to channel order, same as model.
        if full_config is None:
            raise AttributeError("UniEdgeCodecTrainingModule requires full_config")
        encode_signals = list(full_config.dataset.arguments.encode_signals)
        st_map = dict(parameters.pop("signal_type_map"))
        parameters["signal_type_ids"] = [int(st_map[s]) for s in encode_signals]
        return module_mapping[name](model=model, **parameters)

    # Other modules
    else:
        return module_mapping[name](model=model, **parameters)
    

def optimizer_factory(
    name: str,
    parameters: dict,
) -> partial[optim.Optimizer]:
    if name == "adam":
        return partial(optim.Adam, **parameters)
    elif name == "adamw":
        return partial(optim.AdamW, **parameters)
    else:
        raise AttributeError(f"Selected optimizer type {name} is not supported, yet!")


def scheduler_factory(
    function: dict[str, dict], trainer_parameters: dict[str, Any]
) -> partial[optim.lr_scheduler.LRScheduler]:
    scheduler_dict = dict(**trainer_parameters)

    if function["name"] == "cosine_annealing":
        scheduler_dict["scheduler"] = partial(
            optim.lr_scheduler.CosineAnnealingLR, **function["parameters"]
        )
    elif function["name"] == "cosine_annealing_warm_restarts":
        scheduler_dict["scheduler"] = partial(
            optim.lr_scheduler.CosineAnnealingWarmRestarts, **function["parameters"]
        )
    elif function["name"] == "reduce_on_plateau":
        scheduler_dict["scheduler"] = partial(
            optim.lr_scheduler.ReduceLROnPlateau, **function["parameters"]
        )
    else:
        raise AttributeError(
            f"Selected scheduler type {function['name']} is not supported, yet!"
        )

    def scheduler_init(optimizer, scheduler_dict_):
        scheduler_dict_["scheduler"] = scheduler_dict["scheduler"](optimizer=optimizer)
        return scheduler_dict_

    return partial(scheduler_init, scheduler_dict_=scheduler_dict)


def callback_factory(
    callbacks: Union[dict[str, dict[str, Any]], None],
) -> Union[list[pl_callbacks.Callback], None]:
    """Create PyTorch Lightning callbacks from configuration."""

    if callbacks is not None and len(callbacks) > 0:

        supported_callbacks = {
            "earlystopping": pl_callbacks.EarlyStopping,
            "modelcheckpoint": pl_callbacks.ModelCheckpoint,
            "modelsummary": pl_callbacks.ModelSummary,
            "learningratemonitor": pl_callbacks.LearningRateMonitor,
        }

        callback_list = []
        for callback_name, callback_params in callbacks.items():
            if callback_name in supported_callbacks.keys():
                if callback_params is None:
                    callback_params = {}

                callback_list.append(
                    supported_callbacks[callback_name](**callback_params)
                )
            else:
                raise AttributeError(
                    f"Selected callback type {callback_name} is not supported, yet!"
                )
        return callback_list
    return None


def runner_factory(name: str, parameters: dict):
    """
    Create a runner with the given parameters.

    Args:
        name: The type of runner to create
        parameters: Configuration parameters for the runner

    Returns:
        Configured runner instance
    """
    if name == "base":
        return BaseRunner(**parameters)
    else:
        raise AttributeError(
            f"Selected runner type '{name}' is not supported. "
            f"Supported types: 'base'"
        )
