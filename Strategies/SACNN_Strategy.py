import torch
import torch.optim as optim
from tqdm import tqdm
from Strategies.Classification_Strategy import ClassificationStrategy
from Networks import SACNN
from Dataset import StlToPointCloud, Pointnet_Dataset
import os
from torch.utils.data import DataLoader
from utils import WandbLogger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import wandb

class SACNNStrategy(ClassificationStrategy):
    def __init__(self, num_classes, k=25, num_points=1024):
        self.model = SACNN(num_classes=num_classes, k=k)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_points = num_points
        self.output_dir = None

    def prepare_data(self, dataset_path, data_raw=True, train_test_split=0.8):
        
        if data_raw:
            self.output_dir = os.path.join("Data_prepared",f"{os.path.basename(dataset_path)}_{self.num_points}_points")
            print(f"Creating Dataset in Path {self.output_dir}")
            StlToPointCloud(dataset_path=dataset_path, number_of_points=self.num_points, train_test_split=train_test_split)
            dataset_train = Pointnet_Dataset(root_dir=self.output_dir, process_data=True, split="train")
            dataset_test = Pointnet_Dataset(root_dir=self.output_dir, process_data=False, split="test")
        else:
            self.output_dir = dataset_path
            dataset_train = Pointnet_Dataset(root_dir=dataset_path, process_data=False, split="train")
            dataset_test = Pointnet_Dataset(root_dir=dataset_path, process_data=False, split="test")

        return dataset_train, dataset_test  

    def train(self, dataset_train, dataset_val, epochs=10, lr=0.001, batch_size=24, num_workers=8, persistent_workers=True, wandb_project_name="3d_classification", wandb_run_name=None):
        # Init the dataloaders
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        

        self.save_path = os.path.join("results", "SACNN_" + os.path.basename(self.output_dir))
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

        self.model.train()
        best_accuracy = 0
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}:")
            correct = 0
            total = 0
            for batch_id, (batch_data, batch_labels) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):

                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device).long()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            train_accuracy = 100 * correct / total
            print(f"Train Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.2f}%")
            # Log training metrics
            self.logger.log_metrics({
                "train_loss": loss / len(dataloader_train),
                "train_accuracy": train_accuracy,
                "epoch": epoch + 1,
            })

            # Validate and log validation metrics
            val_accuracy, val_loss, all_preds, all_labels = self.eval(dataloader_val)
            self.logger.log_metrics({
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1,
            })

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                    
                self.save(os.path.join(self.save_path, "best_sacnn_model.pth"))

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

            scheduler.step()

        print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
        wandb.finish()

    def eval(self, dataloader_val):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0  # Track total validation loss
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(dataloader_val):
                # Move data and labels to the appropriate device
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device).long()

                # Forward pass
                outputs = self.model(batch_data)

                # Calculate loss for the batch
                loss = self.criterion(outputs, batch_labels)
                total_loss += loss.item()

                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                all_preds.append(predicted.cpu())
                all_labels.append(batch_labels.cpu())

                # Update metrics
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        # Calculate average loss and accuracy
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader_val)

        print(f"Validation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_loss:.4f}")

        return accuracy, avg_loss, torch.cat(all_preds), torch.cat(all_labels)

    def test(self, dataset_test):
        dataloader_val = DataLoader(dataset_test, batch_size=24, shuffle=False)
        self.eval(dataloader_val)

    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=False))
        print(f"Model loaded from {path}")
