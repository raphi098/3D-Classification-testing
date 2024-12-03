import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from Strategies.Classification_Strategy import ClassificationStrategy
from Networks import Pointnet2, Pointnet2_loss
from Dataset import StlToPointCloud, PointCloudDataset
import os
from torch.utils.data import DataLoader
from utils import WandbLogger, Augmentation
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import wandb
from types import SimpleNamespace
from Dataset import PointnetDataset

class Pointnet2Strategy(ClassificationStrategy):
    def __init__(self, num_classes, num_points=1024, use_normals = True, use_uniform_sample = True, unit_ball = True):
        self.model = Pointnet2(num_classes=num_classes, normal_channel=use_normals)
        self.criterion = Pointnet2_loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_points = num_points
        self.output_dir = None
        self.Augmentation = Augmentation()
        self.num_classes = num_classes
        self.use_normals =  use_normals
        self.use_uniform_sample = use_uniform_sample
        self.unit_ball = unit_ball

    def prepare_data(self, dataset_path, data_raw=True, train_test_split=0.8):
        args = SimpleNamespace(use_uniform_sample=self.use_uniform_sample, use_normals=self.use_normals, num_category=self.num_classes)
        if data_raw:
            self.output_dir = os.path.join("Data_prepared", f"{os.path.basename(dataset_path)}_{self.num_points}_points_unitball_{self.unit_ball}")
            print(f"Creating Dataset in Path {self.output_dir}")
            StlToPointCloud(dataset_path=dataset_path, number_of_points=self.num_points, train_test_split=train_test_split, unit_ball=self.unit_ball)
            dataset_train = PointnetDataset(root=self.output_dir, args=args, num_points=self.num_points,process_data=True, split="train")
            dataset_test = PointnetDataset(root=self.output_dir, args=args, num_points=self.num_points, process_data=True, split="test")
        else:
            self.output_dir = dataset_path
            dataset_train = PointnetDataset(root=dataset_path, args=args,num_points=self.num_points, process_data=False, split="train")
            dataset_test = PointnetDataset(root=dataset_path, args=args,num_points=self.num_points, process_data=False, split="test")

        return dataset_train, dataset_test  

    def train(
        self, dataset_train, dataset_val, epochs=100, lr=0.001, batch_size=24, num_workers=8, persistent_workers=True, wandb_project_name="3d_classification", wandb_run_name=None):
        # Init the dataloaders
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True, persistent_workers=persistent_workers)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)
        
        self.model.apply(inplace_relu)
        self.model.to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

        self.save_path = os.path.join("results", f"Pointnet2_{os.path.basename(self.output_dir)}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            os.makedirs(os.path.join(self.save_path, "confusion_matrices"))

        # Dynamically initialize the WandbLogger
        self.logger = WandbLogger(
            project_name=wandb_project_name,
            run_name=wandb_run_name,
            config={
                "num_classes": len(dataloader_train.dataset.classes),
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
            }
        )

        best_accuracy = 0
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}:")
            mean_correct = []
            self.model.train()
            
            for batch_id, (points, target) in tqdm(enumerate(dataloader_train), total=len(dataloader_train), smoothing=0.9):
                #Not good practice to use inplace operations in pytorch, but it is used in the original implementation
                points = points.data.numpy()
                points = self.Augmentation.random_point_dropout(points)
                points[:, :, 0:3] = self.Augmentation.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = self.Augmentation.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()

                pred, trans_feat = self.model(points)
                loss = self.criterion(pred, target.long())
                pred_choice = pred.data.max(1)[1]

                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            train_instance_acc = np.mean(mean_correct) * 100
            print(f"Train Loss: {loss.item():.4f}, Accuracy: {train_instance_acc:.2f}%")
            
            # Log training metrics
            self.logger.log_metrics({
                "train_loss": loss.item(),
                "train_accuracy": train_instance_acc,
                "epoch": epoch + 1,
            })

            # # Validate and log validation metrics
            val_accuracy, val_loss, all_preds, all_labels = self.eval(dataloader_val)
            self.logger.log_metrics({
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1,
            })

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

                self.save(os.path.join(self.save_path, "best_pointnet2_model.pth"))

                # Create and save confusion matrix
                cm = confusion_matrix(all_labels, all_preds, labels=range(len(dataloader_train.dataset.classes)))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataloader_train.dataset.classes)

                # Adjust figure size based on the number of classes
                num_classes = len(dataloader_train.dataset.classes)
                fig_size = max(8, num_classes // 2)  # Dynamically scale figure size, min size of 8
                plt.figure(figsize=(fig_size, fig_size))

                # Plot confusion matrix
                ax = plt.gca()
                disp.plot(cmap=plt.cm.Blues, ax=ax)

                # Rotate x-axis labels for better readability
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

                # Set title and save the confusion matrix plot
                plt.title(f"Confusion Matrix (Epoch {epoch + 1})")
                plt.tight_layout()  # Ensures everything fits within the figure
                plt.savefig(os.path.join(self.save_path, "confusion_matrices", f"confusion_matrix_epoch_{epoch + 1}.png"), bbox_inches="tight")
                plt.close()


        print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
        wandb.finish()

    def eval(self, dataloader_val):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(dataloader_val):
                # Move data and labels to the appropriate device
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device).long()
                batch_data = batch_data.transpose(2, 1)

                # Forward pass
                pred, _ = self.model(batch_data)
                loss = self.criterion(pred, batch_labels)
                total_loss += loss.item()

                # Get predictions
                pred_choice = pred.data.max(1)[1]
                all_preds.append(pred_choice.cpu())
                all_labels.append(batch_labels.cpu())

                # Update total and correct counts
                correct += (pred_choice == batch_labels).sum().item()
                total += batch_labels.size(0)

        # Calculate metrics
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader_val)

        print(f"Validation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_loss:.4f}")
        return accuracy, avg_loss, torch.cat(all_preds), torch.cat(all_labels)

    def test(self, dataset_test):
        dataloader_test = DataLoader(dataset_test, batch_size=24, shuffle=False)
        return self.eval(dataloader_test)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=False))
        print(f"Model loaded from {path}")

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

