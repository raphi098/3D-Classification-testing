from abc import ABC, abstractmethod

class ClassificationStrategy(ABC):
    @abstractmethod
    def prepare_data(self, dataset_path, number_of_points, data_raw=True, train_test_split=0.8):
        """Prepare the input data based on the network's requirements."""
        pass

    @abstractmethod
    def train(self, dataloader_train, dataloader_test):
        """Train the model using the prepared data."""
        pass

    @abstractmethod
    def predict(self, prepared_data):
        """Predict results using the trained model."""
        pass
    
    @abstractmethod
    def test(self, dataloader_test):
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
    
