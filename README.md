# 🧠 Digit Recognition using CNN | MNIST + PyTorch

This project uses a **Convolutional Neural Network (CNN)** built with PyTorch to classify handwritten digits from the MNIST dataset. It includes training, evaluation, custom image prediction, and performance analysis.

---

## 📂 Files & Folders

- `train.py` — Full training + evaluation pipeline.
- `test_images/` — Custom digit images tested on the model.
- `report/digit_recognition_report.pdf` — Final LaTeX project report (includes performance analysis and improvements).
- `requirements.txt` — All dependencies.

---

## 🧪 Sample Output

![Sample Digit](test_images/4.png)

**Predicted Label**: `4`

---

## 📈 Model Accuracy

- ✅ **Train Accuracy**: ~99%
- ✅ **Validation Accuracy**: ~98%
- ✅ **Test Accuracy**: ~98%
- 📊 Confusion matrix & performance visualized in the report.

---

## 📑 Project Report (PDF)

📥 [Click to View Report](https://github.com/arjunsingh196/digit-recognition-cnn/raw/main/report/digit_recognition_report.pdf)

---

## ⚙️ Installation

```bash
git clone https://github.com/arjunsingh196/digit-recognition-cnn.git
cd digit-recognition-cnn
pip install -r requirements.txt
python train.py
