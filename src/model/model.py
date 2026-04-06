import torch
import torch.nn as nn
from torchvision import models
import math


def init_cnn_weights(m):
    """
    Weight initialization helper for custom CNN layers.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Model:
    """
    Wrapper class for PyTorch model to standardize interface.
    """

    def __init__(self, num_classes: int, device: str = "cpu", model_type: str = "resnet18") -> None:
        self.device = torch.device(device)
        self.model = None
        self.model_type = model_type
        self.initialize(num_classes)

    def initialize(self, num_classes: int):
        """
        Initialize the underlying model. Supports 'resnet18' and 'custom_cnn'.
        """
        if self.model_type == "resnet18":
            model = models.resnet18(weights=None)  # Start with untrained weights
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif self.model_type == "custom_cnn":
            # Small custom CNN suitable for 28x28 inputs (MedMNIST)
            class CustomCNN(nn.Module):
                def __init__(self, num_classes: int):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                    )
                    # after two 2x pools, 28x28 -> 7x7
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(64 * 7 * 7, 128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        nn.Linear(128, num_classes),
                    )

                def forward(self, x):
                    x = self.features(x)
                    x = self.classifier(x)
                    return x

            model = CustomCNN(num_classes=num_classes)
            # apply module-level initializer
            model.apply(init_cnn_weights)
        else:
            # fallback to resnet18 if unknown type
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        self.model = model.to(self.device)

    def train(self, dataloader, optimizer, criterion):
        self.model.train()

        total_loss = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.squeeze().to(self.device)

            outputs = self.model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict_proba(self, dataloader):
        self.model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        return all_probs, all_labels

    def predict(self, dataloader):
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        return all_preds, all_labels

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()