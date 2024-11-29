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
from utils import WandbLogger, MultiViewSampler
import wandb


class Rotationnet_Strategy(ClassificationStrategy):
    def __init__(self, num_classes, num_views=12, feature_extractor="alexnet"):
        self.feature_extractor = feature_extractor
        self.num_views = num_views
        self.num_classes = num_classes
        self.output_dir = None
        self.criterion = nn.CrossEntropyLoss()

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

    def prepare_data(self, dataset_path, data_raw=True, train_test_split=0.8):
        """Prepare data by creating multiview datasets."""
        
        if data_raw:
            self.output_dir = os.path.join("Data_prepared", f"{os.path.basename(dataset_path)}_{self.num_views}_views")
            MultiviewDataset(dataset_path, num_views=self.num_views, train_test_split=train_test_split)
        else:
            self.output_dir = dataset_path

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=[0.485], std=[0.229])

        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.output_dir, "train"),
            transform=transforms.Compose(
                [ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
            ),
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(self.output_dir, "val"),
            transform=transforms.Compose(
                [ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
            ),
        )

        return train_dataset, test_dataset

    def train(
        self, dataset_train, dataset_val, epochs=10, lr=0.001, batch_size=24, num_workers=8, persistent_workers=True, wandb_project_name="3d_classification", wandb_run_name=None
    ):
        train_sampler = MultiViewSampler(dataset_size=len(dataset_train), nview=self.num_views)

        dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        )
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers)

        self.save_path = os.path.join("results", f"Pointnet2_{os.path.basename(self.output_dir)}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            os.makedirs(os.path.join(self.save_path, "confusion_matrices"))

        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, self.model.parameters()),  # Only finetunable params
        lr=lr,  # Learning rate might be needed to use 0.001
        betas=(0.9, 0.999),  # Default Adam hyperparameters
        weight_decay=1e-4  # L2 regularization (same as weight decay in SGD)
         )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

        dataloader_val.dataset.imgs = sorted(dataloader_val.dataset.imgs)
        
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
            total_correct = 0
            total_samples = 0
            total_loss = 0.0

            with tqdm(dataloader_train, desc=f"Epoch {epoch + 1}", unit="batch") as pbar:
                for i, (input, target) in enumerate(pbar):
                    nsamp = int(input.size(0) / self.num_views)

                    # Move inputs and targets to GPU
                    input_var = input.cuda()
                    target_ = torch.LongTensor(target.size(0) * self.num_views).cuda()

                    # Forward pass
                    output = self.model(input_var)
                    num_classes = int(output.size(1) / self.num_views) - 1
                    output = output.view(-1, num_classes + 1)

                    # Log softmax and adjust scores for "incorrect view" label
                    output_ = torch.nn.functional.log_softmax(output, dim=1)
                    output_ = output_[:, :-1] - torch.t(output_[:, -1].repeat(1, output_.size(1) - 1).view(output_.size(1) - 1, -1))

                    # Reshape output and compute scores
                    output_ = output_.view(-1, self.num_views * self.num_views, num_classes)
                    output_ = output_.detach().cpu().numpy()  # Detach the tensor before calling numpy()
                    output_ = output_.transpose(1, 2, 0)
                    # initialize target labels with "incorrect view label"
                    for j in range(target_.size(0)):
                        target_[ j ] = num_classes

                    scores = np.zeros((self.vcand.shape[0], num_classes, nsamp))

                    for j in range(self.vcand.shape[0]):
                        for k in range(self.vcand.shape[1]):
                            scores[ j ] = scores[ j ] + output_[ self.vcand[ j ][ k ] * self.num_views + k ]

                    # Adjust targets based on best pose
                    for n in range(nsamp):
                        j_max = np.argmax(scores[:, target[n * self.num_views], n])
                        for k in range(self.vcand.shape[1]):
                            target_[n * self.num_views * self.num_views + self.vcand[j_max][k] * self.num_views + k] = target[n * self.num_views]
                    
                    target_var = target_.cuda()
                    loss = criterion(output, target_var)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Update tqdm progress bar
                    pbar.set_postfix({
                        "loss": total_loss / (i + 1),
                    })
                scheduler.step()


            # Log training metrics
            self.logger.log_metrics({
                "train_loss": loss.item(),
                "epoch": epoch + 1,
            })

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
        all_preds = []
        all_labels = []
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        with torch.no_grad():
            for i, (input, target) in enumerate(dataloader_val):
                # Move data to GPU
                target = target.cuda()
                input_var = input.cuda()

                # Forward pass
                output = self.model(input_var)
                num_classes = int(output.size(1) / self.num_views) - 1
                output = output.view(-1, self.num_views, num_classes + 1)  # Reshape for multi-view structure

                # Apply log-softmax and exclude "incorrect view" label
                output = torch.nn.functional.log_softmax(output, dim=2)
                class_logits = output[:, :, :-1]  # Exclude "incorrect view" column

                # Aggregate scores across views
                class_scores = class_logits.sum(dim=1)  # Shape: (batch_size, num_classes)
                preds = torch.argmax(class_scores, dim=1)  # Predicted classes, shape: (batch_size,)

                # Compute loss (object-level)
                target_var = target[:len(preds)]  # Match target size with preds
                loss = self.criterion(class_scores, target_var)
                total_loss += loss.item()

                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target_var.cpu().numpy())

                # Accuracy calculation
                correct = preds.eq(target_var).sum().item()
                total_correct += correct
                total_samples += len(preds)

        # Compute overall metrics
        accuracy = total_correct / total_samples * 100
        avg_loss = total_loss / len(dataloader_val)

        print(f"Validation Accuracy: {accuracy:.2f}%")
        print(f"Validation Loss: {avg_loss:.4f}")

        return accuracy, avg_loss, all_preds, all_labels


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

def repeat_to_rgb(x):
    return x.repeat(3, 1, 1)
