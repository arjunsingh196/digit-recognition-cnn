# 🧠 Digit Recognition using CNN | MNIST + PyTorch

This project uses a **Convolutional Neural Network (CNN)** built with PyTorch to classify handwritten digits from the MNIST dataset. It includes training, evaluation, custom image prediction, and performance analysis.


---

## 🔍 Model Overview

- 🧱 **Architecture**: 3 convolutional layers + 3 max-pool layers + 3 fully connected layers
- ⚙️ **Optimizer**: Adam
- 🎯 **Loss Function**: CrossEntropyLoss
- 🗃️ **Dataset**: MNIST (60,000 train + 10,000 test images)

---

## 📊 Accuracy Metrics

| Epoch | Loss  | Train Accuracy | Val Accuracy |
|-------|-------|----------------|--------------|
| 1     | 0.718 | 76.04%         | 92.80%       |
| 2     | 0.255 | 94.00%         | 96.00%       |
| 3     | 0.191 | 96.30%         | 97.30%       |
| 4     | 0.159 | 97.14%         | 97.60%       |
| 5     | 0.130 | 97.90%         | 98.10%       |

✅ **Test Accuracy**: 98.35%

---

## 🧪 Predict on Your Own Image

1. Save your digit image (e.g., `4.png`) in `test_images/`
2. In `train.py`, add or use the prediction block:
   ```python
   img = cv2.imread("test_images/4.png")
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   resized = cv2.resize(gray, (28, 28))
   tensor_img = torch.tensor(resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
   output = model(tensor_img)
   pred = torch.argmax(output, 1).item()
   print("Predicted Digit:", pred)
