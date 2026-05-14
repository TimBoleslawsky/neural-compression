from omegaconf import OmegaConf


class TrainingArgs:
    def __init__(self, config_path):
        config = OmegaConf.load(config_path)
        self.dataset = config.dataset
        self.dataloader = config.dataloader
        self.model = config.model
        self.config_path = config_path
        self.mode = config.mode
        self.training_module = config.training_module
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.callbacks = config.callbacks
        self.trainer = config.trainer
        self.checkpoints = config.checkpoints
        self.runner = config.runner
        # Optional: Random seed for reproducibility (defaults to 42)
        self.seed = config.get("seed", 42)
