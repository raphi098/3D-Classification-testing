from Strategies.Classification_Strategy import ClassificationStrategy
from Networks import Rotationnet
import torchvision.models as models
from torchvision.models import AlexNet_Weights, ResNet18_Weights
import numpy as np
import torch
from Dataset import MultiviewDataset
from torchvision import datasets, transforms
import os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import WandbLogger
import wandb


class Rotationnet_Strategy(ClassificationStrategy):
    def __init__(self, num_classes, num_views=12, feature_extractor="alexnet"):
        self.feature_extractor = feature_extractor
        self.num_views = num_views
        self.num_classes = num_classes
        self.output_dir = None

        # Use appropriate weights based on the architecture
        if feature_extractor == "alexnet":
            model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        elif feature_extractor == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.__dict__[feature_extractor](weights="DEFAULT")

        # Load viewpoint candidates
        if num_views == 12:
            self.vcand = np.load(os.path.join("utils", "vcand_case1.npy"))
        elif num_views == 20:
            self.vcand = np.load(os.path.join("utils", "vcand_case2.npy"))
        else:
            raise ValueError("Invalid number of views specified. Must be 12 or 20.")

        self.model = Rotationnet(
            original_model=model, arch=feature_extractor, num_classes=(num_classes + 1) * num_views
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def prepare_data(self, dataset_path, data_raw=True, train_test_split=0.8):
        """Prepare data by creating multiview datasets."""
        
        if data_raw:
            self.output_dir = os.path.join("Data_prepared", f"{os.path.basename(dataset_path)}_{self.num_views}_views")
            MultiviewDataset(dataset_path, num_views=self.num_views, train_test_split=train_test_split)
        else:
            self.output_dir = dataset_path

        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.output_dir, "train"),
            transform=transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
            ),
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(self.output_dir, "val"),
            transform=transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
            ),
        )

        return train_dataset, test_dataset

    def train(self, dataset_train, dataset_val, epochs=10, lr=0.001, batch_size=24, num_workers=8, persistent_workers=True, wandb_project_name="3d_classification", wandb_run_name=None):
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)

        self.save_path = os.path.join("results", f"{self.feature_extractor}_{os.path.basename(self.output_dir)}")
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        criterion = nn.CrossEntropyLoss().to(self.device)

        # Dynamically initialize the WandbLogger
        self.logger = WandbLogger(
            project_name=wandb_project_name,
            run_name=wandb_run_name,
            config={
                "num_classes": self.num_classes,
                "num_views": self.num_views,
                "feature_extractor": self.feature_extractor,
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
            }
        )

        best_accuracy = 0
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (batch_data, batch_labels) in enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{epochs}")):
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

            train_accuracy = 100 * correct / total
            avg_loss = running_loss / len(dataloader_train)

            # Log training metrics
            self.logger.log_metrics({
                "train_loss": avg_loss,
                "train_accuracy": train_accuracy,
                "epoch": epoch + 1,
            })

            # Validate and log validation metrics
            val_accuracy, all_preds, all_labels = self.eval(dataloader_val)
            self.logger.log_metrics({
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1,
            })

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                    os.makedirs(os.path.join(self.save_path, "confusion_matrices"))
                self.save(os.path.join(self.save_path, f"best_rotationnet_model_epoch_{epoch + 1}.pth"))

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
        """Evaluate the model on the validation set."""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(dataloader_val, desc="Evaluating"):
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                outputs = self.model(batch_data)
                _, preds = torch.max(outputs, 1)

                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(batch_labels.cpu())

        accuracy = 100 * correct / total
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy, all_preds, all_labels

    def test(self, dataset_test):
        """Test the model on unseen data."""
        test_loader = DataLoader(dataset_test, batch_size=24, shuffle=False)
        accuracy = self.eval(test_loader)
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def save(self, path):
        """Save the model."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load the model."""
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
