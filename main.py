from Strategies import SACNNStrategy, PointnetStrategy, Pointnet2Strategy, Rotationnet_Strategy
from AIWorkflow import AIWorkflow
import os

if __name__ == "__main__":
    # Instantiate the strategy and workflow
    strategy = Rotationnet_Strategy(num_classes=2)
    workflow = AIWorkflow(strategy)

    # Define the path to the data
    path_data = os.path.join(os.getcwd(), "Data_raw", "test_dataset")

    # Prepare the data
    dataset_train, dataset_test = workflow.prepare_data(path_data, data_raw=True, train_test_split=0.8)

    workflow.run_training(dataset_train, dataset_test)