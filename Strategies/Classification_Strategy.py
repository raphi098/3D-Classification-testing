from abc import ABC, abstractmethod

class ClassificationStrategy(ABC):
    @abstractmethod
    def prepare_data(self, dataset_path, number_of_points, data_raw=True, train_test_split=0.8):
        """Prepare the input data based on the network's requirements."""
        pass

    @abstractmethod
    def train(self, dataset_train, dataset_val, epochs=10, lr=0.001, batch_size=24, num_workers=8, persistent_workers=True, wandb_project_name="3d_classification", wandb_run_name=None):
        """Train the model using the prepared data."""
        pass
    
    @abstractmethod
    def eval(self, dataloader_val):
        """Test the model using the prepared data."""
        pass
    
    @abstractmethod
    def test(self, dataset_test):
        """Test the model using the prepared data."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the model to the specified path."""
        pass
    @abstractmethod
    def load(self, path):
        """Load the model from the specified path."""
        pass 
    
