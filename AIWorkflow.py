class AIWorkflow:
    def __init__(self, strategy):
        """
        Initialize the workflow with a specific classification strategy.

        Args:
            strategy (ClassificationStrategy): An instance of a strategy class.
        """
        self.strategy = strategy

    def prepare_data(self, dataset_path, data_raw=True, train_test_split=0.8):
        """
        Prepare the dataset using the selected strategy.

        Args:
            dataset_path (str): Path to the dataset directory.
            number_of_points (int): Number of points to sample from each STL file. Leave with None for multiview datasets.
            data_raw (bool): Whether to process raw STL data.
            train_test_split (float): Ratio for splitting the dataset into training and testing.

        Returns:
            Tuple[DataLoader, DataLoader]: Dataloaders for training and testing.
        """
        print(f"Preparing data using strategy {self.strategy.__class__.__name__}")

        return self.strategy.prepare_data(dataset_path, data_raw, train_test_split)

    def run_training(self, dataset_train, dataset_val, epochs=100, lr=0.001, batch_size=24, num_workers=8, persistent_workers=True, wandb_project_name="3d_classification", wandb_run_name=None):
        """
        Run the training process using the selected strategy.

        Args:
            dataset_train (Dataset): Training dataset.
            dataset_val (Dataset): Validation dataset.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            batch_size (int, optional): Batch size for training and validation. Defaults to 24.
            num_workers (int, optional): Number of workers for data loading. Defaults to 8.
            persistent_workers (bool, optional): Whether to keep data loading workers alive between epochs. Defaults to True.
            wandb_project_name (str, optional): Project name for logging metrics to Weights and Biases. Defaults to "3d_classification".
            wandb_run_name (str, optional): Run name for logging to Weights and Biases. Defaults to None.

        Returns:
            None
        """
        self.strategy.train(dataset_train, dataset_val, epochs, lr, batch_size, num_workers, persistent_workers, wandb_project_name, wandb_run_name)


    def test_model(self, dataloader_test):
        """
        Test the model using the selected strategy.

        Args:
            dataloader_test (DataLoader): Testing dataset loader.

        Returns:
            Tuple[float, Tensor, Tensor]: Test accuracy, predictions, and ground truth labels.
        """
        return self.strategy.test(dataloader_test)

    def save_model(self, path):
        """
        Save the trained model to the specified path using the selected strategy.

        Args:
            path (str): Path to save the model.
        """
        self.strategy.save(path)

    def load_model(self, path):
        """
        Load a model from the specified path using the selected strategy.

        Args:
            path (str): Path to the saved model.
        """
        self.strategy.load(path)
