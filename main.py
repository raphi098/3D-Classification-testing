from Strategies import SACNNStrategy, PointnetStrategy, Pointnet2Strategy, Rotationnet_Strategy
from AIWorkflow import AIWorkflow
import os

if __name__ == "__main__":
    # Define the path to the data
    # path_data = os.path.join(os.getcwd(), "Data_raw", "1gliedrig_100_files")
    path_data = os.path.join(os.getcwd(), "Data_prepared", "1gliedrig_100_files_1024_points")
    # Instantiate the strategy and workflow
    strategy = SACNNStrategy(num_classes=9)
    workflow = AIWorkflow(strategy)
    dataset_train, dataset_test = workflow.prepare_data(path_data, data_raw=False)
    workflow.run_training(dataset_train, dataset_test, wandb_run_name="SACNN", epochs=100)


    # strategy = PointnetStrategy(num_classes=9)
    # workflow = AIWorkflow(strategy)
    # dataset_train, dataset_test = workflow.prepare_data(path_data, data_raw=False)
    # workflow.run_training(dataset_train, dataset_test, wandb_run_name="Pointnet", epochs=100)

    # strategy = Pointnet2Strategy(num_classes=9)
    # workflow = AIWorkflow(strategy)
    # dataset_train, dataset_test = workflow.prepare_data(path_data, data_raw=False)
    # workflow.run_training(dataset_train, dataset_test, wandb_run_name="Pointnet2", epochs=100)

    # # Instantiate the strategy and workflow 
    # strategy = Rotationnet_Strategy(num_classes=9, feature_extractor="resnet18")
    # workflow = AIWorkflow(strategy)

    # # Prepare the data
    # dataset_train, dataset_test = workflow.prepare_data(path_data, data_raw=False)
    # workflow.run_training(dataset_train, dataset_test, wandb_run_name="Rotationnet_resnet18_12views")

    # # Instantiate the strategy and workflow 
    # strategy = Rotationnet_Strategy(num_classes=9)
    # workflow = AIWorkflow(strategy)
    # dataset_train, dataset_test = workflow.prepare_data(path_data, data_raw=False)
    # workflow.run_training(dataset_train, dataset_test, wandb_run_name="Rotationnet_alexnet_12views")

    # # Instantiate the strategy and workflow
    # strategy = Rotationnet_Strategy(num_classes=9, feature_extractor="resnet18", num_views=20)
    # workflow = AIWorkflow(strategy)
    # dataset_train, dataset_test = workflow.prepare_data(path_data)
    # workflow.run_training(dataset_train, dataset_test, wandb_run_name="Rotationnet_resnet18_20views")

    # # Instantiate the strategy and workflow
    # strategy = Rotationnet_Strategy(num_classes=9, num_views=20)
    # workflow = AIWorkflow(strategy)
    # dataset_train, dataset_test = workflow.prepare_data(path_data, data_raw=False)
    # workflow.run_training(dataset_train, dataset_test, wandb_run_name="Rotationnet_alexnet_20views")


    
    
    
