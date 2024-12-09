from Strategies import SACNNStrategy, PointnetStrategy, Pointnet2Strategy, Rotationnet_Strategy
from AIWorkflow import AIWorkflow
import os

if __name__ == "__main__":
    path_data = os.path.join(os.getcwd(), "Data_prepared", "1gliedrig_50_files_unitball_True_points_1024")

    strategy = Pointnet2Strategy(num_classes=13, num_points=1024)
    workflow = AIWorkflow(strategy)
    dataset_train, dataset_test = workflow.prepare_data(path_data, data_raw=False)

    # workflow.run_training(dataset_train, dataset_test, epochs=150, lr=0.001, batch_size=24, num_workers=8, persistent_workers=True, wandb_project_name="3d_classification", 
    #                       wandb_run_name="all_1024")


    # Testing
    workflow.load_model(os.path.join("results","Pointnet2_msg_1gliedrig_50_files_points_1024","best_pointnet2_model.pth"))
    val_accuracy, val_loss, all_preds, all_labels = workflow.test_model(dataset_test)

    # Ensure predictions and labels are in the same format (e.g., tensors or lists)
    if not isinstance(all_preds, list):
        all_preds = all_preds.tolist()
    if not isinstance(all_labels, list):
        all_labels = all_labels.tolist()

    # Find mismatched indices
    mismatched_indices = [
        idx for idx, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label
    ]
    # Print mismatched indices
    print(f"Mismatched indices: {mismatched_indices}")
    print(f"Total mismatches: {len(mismatched_indices)}")

    # Optionally, log mismatched predictions and labels
    for idx in mismatched_indices:
        print(f"Index: {idx}, Prediction: {all_preds[idx]}, Ground Truth: {all_labels[idx]}")

    with open(os.path.join("Data_prepared", "1gliedrig_50_files_unitball_True_points_1024", "test.txt" )) as f:
        file = f.readlines()
    with open(os.path.join("Data_prepared", "1gliedrig_50_files_unitball_True_points_1024", "shape_names.txt" )) as f:
        shape_names = f.readlines()
    mismatches_filenames = [file[idx] for idx in mismatched_indices]
    for idx in mismatched_indices:
        print(f"File: {file[idx]}",  "Prediction: ", shape_names[all_preds[idx]])
