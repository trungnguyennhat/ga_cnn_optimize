"""CNN baseline for 64x64 RGB DermaMNIST images."""

from torch import nn


class BaselineCNN(nn.Module):
    def __init__(self, filters: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        if filters <= 0:
            raise ValueError("filters must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.features = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters, filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters * 2, filters * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(filters * 4 * 8 * 8, 7),
        )

    def forward(self, images):
        return self.classifier(self.features(images))
