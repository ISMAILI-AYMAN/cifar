import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from cifarmodel import SimpleCNN  # ← import your model class
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)

# Initialize model, loss, optimizer
net = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop
train_losses = []

for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            avg_loss = running_loss / 100
            train_losses.append(avg_loss)
            print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {avg_loss:.3f}")
            running_loss = 0.0

import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.title("Training Loss Over Time")
plt.xlabel("Batch (per 100 steps)")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

print("✅ Training complete")

# Load test data
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"🎯 Test Accuracy: {100 * correct / total:.2f}%")
torch.save(net.state_dict(), "cnn_cifar10.pt")
print("💾 Model saved as cnn_cifar10.pt")
model = SimpleCNN()
model.load_state_dict(torch.load("cnn_cifar10.pt"))
model.eval()
