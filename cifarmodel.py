import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # Input: 3x32x32 → Output: 16x32x32
        self.pool = nn.MaxPool2d(2, 2)               # Output: 16x16x16
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # Output: 32x16x16 → 32x8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)                # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = x.view(-1, 32 * 8 * 8)            # Flatten
        x = F.relu(self.fc1(x))               # Fully connected
        x = self.fc2(x)                       # Output layer (logits)
        return x


