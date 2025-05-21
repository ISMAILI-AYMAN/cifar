import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from cifarmodel import SimpleCNN
import numpy as np
modelpath = "cnn_cifar10.pt"  # Path to your saved model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load CIFAR-10 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False)

# Load your model
net = SimpleCNN()
net.load_state_dict(torch.load(modelpath, map_location=device))
net.to(device)
net.eval()

# ----------- Sample Predictions -------------
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

outputs = net(images)
_, predicted = torch.max(outputs, 1)

images = images.cpu()
labels = labels.cpu()
predicted = predicted.cpu()

fig = plt.figure(figsize=(10, 4))
for i in range(5):
    ax = fig.add_subplot(1, 5, i+1)
    img = images[i] / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(f"Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# ----------- Confusion Matrix -------------
y_true, y_pred = [], []

with torch.no_grad():
    for data in testloader:
        imgs, lbls = data[0].to(device), data[1].to(device)
        outs = net(imgs)
        _, preds = torch.max(outs, 1)
        y_true.extend(lbls.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()
print("✅ Evaluation complete")