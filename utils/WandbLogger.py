import wandb
import os

class WandbLogger:
    def __init__(self, project_name, entity_name="raphaeldechent1-technische-hochschule-augsburg", config=None, run_name=None):
        """
        Initialize the WandbLogger.
        Args:
            project_name (str): The W&B project name.
            entity_name (str): The W&B team or username (optional).
            config (dict): Dictionary of configuration parameters to log.
            run_name (str): Optional name for the W&B run.
        """
        self.run = wandb.init(
            project=project_name,
            entity=entity_name,
            config=config,
            name=run_name
        )

    def log_metrics(self, metrics, step=None):
        """
        Log metrics to W&B.
        Args:
            metrics (dict): A dictionary of metrics to log.
            step (int): Optional step or epoch number.
        """
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def save_model(self, path, epoch=None):
        """
        Save the model and log the checkpoint to W&B.
        Args:
            model: The PyTorch model to save.
            path (str): Path to save the model.
            epoch (int): Optional epoch number to include in the file name.
        """
        wandb.save(path)
        print(f"Model checkpoint saved and logged to W&B: {path}")

    def finish(self):
        """
        Finish the W&B run.
        """
        wandb.finish()
