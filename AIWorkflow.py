class AIWorkflow:
    def __init__(self, strategy):
        """
        Initialize the workflow with a specific classification strategy.

        Args:
            strategy (ClassificationStrategy): An instance of a strategy class.
        """
        self.strategy = strategy

    def prepare_data(self, dataset_path, number_of_points = None, data_raw=True, train_test_split=0.8):
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
        if number_of_points is None:
            return self.strategy.prepare_data(dataset_path, data_raw, train_test_split)
        else:
            return self.strategy.prepare_data(dataset_path, number_of_points, data_raw, train_test_split)

    def run_training(self, dataloader_train, dataloader_val, epochs=10, lr=0.001):
        """
        Run the training process using the selected strategy.

        Args:
            dataloader_train (DataLoader): Training dataset loader.
            dataloader_val (DataLoader): Validation dataset loader.
            epochs (int): Number of training epochs.
        """
        self.strategy.train(dataloader_train, dataloader_val, epochs, lr)

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
