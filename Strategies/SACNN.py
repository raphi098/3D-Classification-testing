import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ClassificationStrategy import ClassificationStrategy
from Models.SACNN import SACNN
from DatasetLoader.Stl_to_Pointcloud import StlToPointCloud
from DatasetLoader.SACNNDataLoader import PointCloudDataset
import os

class SACNNStrategy(ClassificationStrategy):
    def __init__(self, num_classes, k=25):
        self.model = SACNN(num_classes=num_classes, k=k)
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def prepare_data(self, dataset_path, number_of_points, data_raw=True, train_test_split=0.8):
        if data_raw:
            new_dataset_path = os.path.join("Data_prepared",f"{os.path.basename(dataset_path)}_{number_of_points}_points")
            print(f"Creating Dataset in Path {new_dataset_path}")
            StlToPointCloud(dataset_path=dataset_path, number_of_points=number_of_points, train_test_split=train_test_split)
            dataloader_train = PointCloudDataset(root_dir=new_dataset_path, process_data=True, split="train")
            dataloader_test = PointCloudDataset(root_dir=new_dataset_path, process_data=False, split="test")
        else:
            dataloader_train = PointCloudDataset(root_dir=dataset_path, process_data=False, split="train")
            dataloader_test = PointCloudDataset(root_dir=dataset_path, process_data=False, split="test")

        return dataloader_train, dataloader_test  # Placeholder, as SACNN expects batch input in training

    def train(self, dataloader_train, dataloader_val, epochs=10, device=None):
        if not device:
            device = self.device
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)
        self.model.train()

        best_accuracy = 0
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}:")
            correct = 0
            total = 0
            for batch_id, (batch_data, batch_labels) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
                self.optimizer.zero_grad()

                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device).long()
                outputs = self.model(batch_data)
                loss = F.cross_entropy(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Train Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

            # Validate
            accuracy, _, _ = self.evaluate(dataloader_val, device)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save("best_sacnn_model.pth")

        print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

    def test(self, dataloader_test, device=None):
        if not device:
            device = self.device

        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(dataloader_test):
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device).long()
                outputs = self.model(batch_data)

                _, predicted = torch.max(outputs.data, 1)
                all_preds.append(predicted.cpu())
                all_labels.append(batch_labels.cpu())

                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy, torch.cat(all_preds), torch.cat(all_labels)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
