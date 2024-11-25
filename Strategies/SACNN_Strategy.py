import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from Strategies.Classification_Strategy import ClassificationStrategy
from Networks import SACNN
from Dataset import StlToPointCloud
from Dataset import PointCloudDataset
import os
from torch.utils.data import DataLoader
from utils import WandbLogger

class SACNNStrategy(ClassificationStrategy):
    def __init__(self, num_classes, k=25, num_points=1024):
        self.model = SACNN(num_classes=num_classes, k=k)
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_points = num_points

    def prepare_data(self, dataset_path, data_raw=True, train_test_split=0.8):
        if data_raw:
            new_dataset_path = os.path.join("Data_prepared",f"{os.path.basename(dataset_path)}_{self.num_points}_points")
            print(f"Creating Dataset in Path {new_dataset_path}")
            StlToPointCloud(dataset_path=dataset_path, number_of_points=self.num_points, train_test_split=train_test_split)
            dataset_train = PointCloudDataset(root_dir=new_dataset_path, process_data=True, split="train")
            dataset_test = PointCloudDataset(root_dir=new_dataset_path, process_data=False, split="test")
        else:
            dataset_train = PointCloudDataset(root_dir=dataset_path, process_data=False, split="train")
            dataset_test = PointCloudDataset(root_dir=dataset_path, process_data=False, split="test")

        return dataset_train, dataset_test  

    def train(self, dataset_train, dataset_val, epochs=10, lr=0.001, batch_size=24, num_workers=8, persistent_workers=True, wandb_project_name="3d_classification", wandb_run_name=None):
        # Init the dataloaders
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)

        # Dynamically initialize the WandbLogger
        self.logger = WandbLogger(
            project_name=wandb_project_name,
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
            epoch_loss = 0.0
            for batch_id, (batch_data, batch_labels) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
                self.optimizer.zero_grad()

                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device).long()
                outputs = self.model(batch_data)
                loss = F.cross_entropy(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            train_accuracy = 100 * correct / total
            print(f"Train Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.2f}%")
            # Log training metrics
            self.logger.log_metrics({
                "train_loss": epoch_loss / len(dataloader_train),
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
                self.save("best_sacnn_model.pth")

        print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

    def eval(self, dataloader_val):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0  # Track total validation loss
        all_preds = []
        all_labels = []

        # Define the loss criterion (e.g., CrossEntropyLoss)
        criterion = F.cross_entropy

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(dataloader_val):
                # Move data and labels to the appropriate device
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device).long()

                # Forward pass
                outputs = self.model(batch_data)

                # Calculate loss for the batch
                loss = criterion(outputs, batch_labels)
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
