import streamlit as st
import torch
from cifarmodel import SimpleCNN
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load model
device = torch.device("cpu")
model = SimpleCNN()
model.load_state_dict(torch.load("cnn_cifar10.pt", map_location=device))
model.eval()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title("CIFAR-10 Classifier (PyTorch + Streamlit)")
uploaded_file = st.file_uploader("Upload a 32x32 image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess & predict
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    st.success(f"Predicted: {classes[predicted.item()]}")
