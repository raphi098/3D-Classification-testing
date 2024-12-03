from Strategies import SACNNStrategy, PointnetStrategy, Pointnet2Strategy, Rotationnet_Strategy
from AIWorkflow import AIWorkflow
import os

if __name__ == "__main__":
    path_data = os.path.join(os.getcwd(), "Data_raw", "all_100_files")

    # strategy = Pointnet2Strategy(num_classes=len(os.listdir(path_data)), num_points=1024, use_normals=False, use_uniform_sample=False)
    # workflow = AIWorkflow(strategy)
    # dataset_train, dataset_test = workflow.prepare_data(path_data, data_raw=True)
    # workflow.run_training(dataset_train, dataset_test, wandb_run_name="Pointnet2_all_unitBall_1024", epochs=100, wandb_project_name="Pointnet2_Finetuning",)

    strategy = Pointnet2Strategy(num_classes=len(os.listdir(path_data)), num_points=2048, use_normals=False, use_uniform_sample=False, unit_ball=False)
    workflow = AIWorkflow(strategy)
    dataset_train, dataset_test = workflow.prepare_data(path_data, data_raw=True)
    workflow.run_training(dataset_train, dataset_test, wandb_run_name="Pointnet2_all_normals_uniform_2048", epochs=100, wandb_project_name="Pointnet2_Finetuning",)

    #Für weitere Tests in pointnet2 strategy process data auf true