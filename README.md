
# CIFAR-10 Image Classifier with PyTorch 🧠🖼️

This project implements a simple Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. The model is trained, evaluated, and visualized with clean modular code.

## 🚀 Features

- CNN architecture built with `torch.nn.Module`
- Training loop with SGD + CrossEntropyLoss
- GPU support via `torch.device`
- Evaluation on test set with accuracy metric
- Visualizations:
  - Training loss over time
  - Sample predictions (correct/wrong)
  - Confusion matrix
- Modular code structure with `model.py`, `train.py`, `evaluate.py`

---

## 🗂️ Project Structure

pytorchproject/
├── day1_load_data/
│ └── cifar_loader.py
├── day2_model/
│ └── model.py
├── day3_training/
│ ├── train.py
│ └── model.py
├── day4_saved_model/
│ └── cnn_cifar10.pt
├── day5_visualize/
│ ├── evaluate.py
│ └── model.py
└── README.md

yaml
Copier
Modifier

---

## 📦 Requirements

```bash
pip install torch torchvision matplotlib scikit-learn
🏃 How to Run
Train the model

bash
Copier
Modifier
cd day3_training
python train.py
Evaluate model + visualize predictions

bash
Copier
Modifier
cd day5_visualize
python evaluate.py
📈 Sample Output
Loss Curve	Confusion Matrix
(Insert screenshot)	(Insert screenshot)

🎯 Final Accuracy
text
Copier
Modifier
✅ Training complete
🎯 Test Accuracy: 50.65%
📌 Author
Ayman Ismaili












#   C I F A R 1 0  
 