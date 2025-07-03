import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# %% Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Extract image tensors and labels
x_train = torch.stack([img for img, _ in train_data])
y_train = torch.tensor([label for _, label in train_data])
x_test  = torch.stack([img for img, _ in test_data])
y_test  = torch.tensor([label for _, label in test_data])

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# %% Visualize Sample Image
plt.imshow(x_train[0].squeeze(), cmap='binary')
plt.title(f"Label: {y_train[0].item()}")
plt.axis('off')
plt.show()

# %% CNN Model Definition
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 1 * 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DigitClassifier()
print(model)

# %% Prepare DataLoaders
train_dataset = TensorDataset(x_train, y_train)
train_size = int(0.7 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_set, val_set = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

print("Train size:", len(train_set))
print("Validation size:", len(val_set))

# %% Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# %% Training Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            _, val_pred = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_pred == val_labels).sum().item()
    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")

# %% Evaluate on Test Set
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"\nTest Accuracy on 10,000 test samples: {test_accuracy:.2f}%")

# %% Image check

import cv2
import numpy as np

# Load image using OpenCV
img = cv2.imread(r'C:\Users\Lenovo\OneDrive\Desktop\4.png')

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize to 28x28 (same as MNIST)
resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

# Normalize the pixel values to [0,1]
normalized = resized / 255.0

# Convert to tensor and add batch and channel dimensions
tensor_img = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 28, 28]
tensor_img = tensor_img.to(device)

# Predict using the PyTorch model
model.eval()
with torch.no_grad():
    output = model(tensor_img)
    pred = torch.argmax(output, 1).item()

print("Predicted digit:", pred)

# %% Final Result
print(f"\nTest Accuracy on 10,000 test samples: {test_accuracy:.2f}%")



# %% Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Re-run inference to collect all predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Display it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(10)])
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix on MNIST Test Set")
plt.show()
