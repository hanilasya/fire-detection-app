# 🔥 Fire Detection System using Deep Learning (CNN)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Accuracy](https://img.shields.io/badge/Accuracy-96.57%25-brightgreen)

A deep learning-based web application that detects **Fire** or **No Fire** in images using a custom Convolutional Neural Network (CNN) built with TensorFlow and Keras, deployed as an interactive web app using Streamlit.

---

## 🌐 Live Demo

🔗 [fire-detection-using-deep-learning-cnn.streamlit.app](https://fire-detection-using-deep-learning-cnn.streamlit.app)

---

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [CNN Architecture](#cnn-architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Results](#results)
- [Applications](#applications)
- [Advantages Over Existing Solutions](#advantages-over-existing-solutions)
- [How It Differs from Traditional Smoke Detectors](#how-it-differs-from-traditional-smoke-detectors)

---

## 📖 Overview

Fire accidents are one of the leading causes of property damage and loss of life worldwide. Early detection is critical to minimizing damage. Traditional fire and smoke detectors rely on physical sensors that require close proximity to the source. This project takes a different approach — using **computer vision and deep learning** to detect fire visually from images, enabling remote, scalable, and faster detection.

This system:
- Accepts any image as input
- Runs it through a trained CNN model
- Predicts whether the image contains **Fire** or **No Fire**
- Displays the result with a **confidence score**

---

## ⚙️ How It Works

```
User uploads image
        ↓
Image resized to 128x128 pixels
        ↓
Pixel values normalized (0 to 1)
        ↓
Fed into trained CNN model
        ↓
Model outputs probability score
        ↓
Score < 0.5  →  🔥 Fire Detected
Score >= 0.5 →  ✅ No Fire
        ↓
Result + Confidence % displayed on screen
```

The entire pipeline runs in **under 1 second** per image.

---

## 🧠 CNN Architecture

The model is a custom Convolutional Neural Network built from scratch using Keras Sequential API.

| Layer | Type | Details |
|---|---|---|
| 1 | Conv2D | 32 filters, 3×3 kernel, ReLU |
| 2 | MaxPooling2D | 2×2 pool size |
| 3 | Conv2D | 64 filters, 3×3 kernel, ReLU |
| 4 | MaxPooling2D | 2×2 pool size |
| 5 | Conv2D | 128 filters, 3×3 kernel, ReLU |
| 6 | MaxPooling2D | 2×2 pool size |
| 7 | Flatten | — |
| 8 | Dense | 128 units, ReLU |
| 9 | Dropout | 50% (prevents overfitting) |
| 10 | Dense (Output) | 1 unit, Sigmoid (binary classification) |

**Why this architecture?**
- Three Conv+Pool blocks progressively extract low-level (edges, textures) to high-level (flame shapes, glow patterns) features
- Dropout layer prevents the model from memorizing training data
- Sigmoid output gives a probability between 0 and 1, perfect for binary classification

**Training Configuration:**
- Loss Function: `binary_crossentropy`
- Optimizer: `Adam`
- Epochs: `10`
- Batch Size: `32`
- Input Size: `128×128×3`

---

## 💻 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.21.0 + Keras 3.14.0 |
| Image Processing | OpenCV, NumPy, Pillow |
| Web App | Streamlit |
| Model Format | Keras Native (`.keras`) |
| Deployment | Streamlit Cloud |
| Version Control | Git + GitHub |

---

## 📦 Dataset

**Dataset:** Fire Detection Dataset by Phylake1337 (NASA Space Apps Challenge 2018)

**Source:** [Kaggle — phylake1337/fire-dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)

| Class | Images |
|---|---|
| Fire (including smoky fire) | 755 |
| No Fire (nature scenes) | 244 |
| **Total** | **~1000** |

**Split:** 80% Train / 20% Test

**Preprocessing:**
- Images resized to 128×128
- Pixel values normalized to [0, 1]
- Horizontal flip augmentation applied during training

> Note: The fire folder contains images with heavy smoke mixed with fire, which makes the model robust to smoky fire conditions as well.

---

## 📁 Project Structure

```
FireSmokeCNN/
│
├── dataset/                  # (local only, not pushed to GitHub)
│   ├── train/
│   │   ├── fire/
│   │   └── no_fire/
│   └── test/
│       ├── fire/
│       └── no_fire/
│
├── model/
│   └── fire_smoke_model.keras  # Trained CNN model
│
├── train.py                  # Model training script
├── predict.py                # CLI prediction script
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.11
- Git

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/satyaprakash-22/fire-detection-using-deep-learning-cnn.git
cd fire-detection-using-deep-learning-cnn
```

**2. Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the web app**
```bash
streamlit run app.py
```

App opens automatically at `http://localhost:8501`

---

### Optional: Retrain the Model

If you want to train from scratch with your own dataset:

```bash
python train.py
```

To test a single image from CLI:
```bash
python predict.py
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **96.57%** |
| Inference Time | < 1 second |
| Model Size | ~38 MB |
| Training Time (CPU) | ~7 minutes |

---

## 🌍 Applications

This system has wide real-world applicability:

- **🏠 Home Security Integration** — Can be integrated with CCTV cameras at home to automatically detect fire and send alerts to homeowners in real time
- **🏭 Industrial Monitoring** — Deploy in factories, warehouses, and power plants where fire risk is high for continuous visual monitoring
- **🌲 Forest Fire Detection** — Mount cameras in forests or use drone footage to detect wildfire in remote areas before it spreads
- **🏫 Smart Buildings** — Integrate with building management systems to trigger sprinklers, alarms, or emergency protocols automatically
- **🚗 Tunnel & Highway Surveillance** — Monitor road tunnels and highways where fire incidents can be catastrophic
- **📱 Mobile Apps** — Embed the model in mobile applications for on-the-go fire detection using phone cameras
- **☁️ Cloud Surveillance Platforms** — Scale the model across thousands of camera feeds in smart city infrastructure

---

## ✅ Advantages Over Existing Solutions

| Feature | Traditional Systems | This CNN System |
|---|---|---|
| Detection method | Physical sensors (heat/smoke) | Visual image analysis |
| Remote detection | ❌ Requires proximity | ✅ Works from any distance |
| Camera integration | ❌ Separate system | ✅ Works directly with cameras |
| False alarm rate | High (dust, steam trigger) | Lower with visual confirmation |
| Scalability | Limited | Highly scalable via cloud |
| Cost | High hardware cost | Low — runs on any machine |
| Real-time capability | Yes | Yes (<1 sec inference) |

---

## 🆚 How It Differs from Traditional Smoke Detectors

Traditional smoke detectors are **physical devices** that:
- Detect smoke particles in the air using ionization or photoelectric sensors
- Must be physically installed in every room
- Only trigger when smoke reaches the sensor directly
- Cannot distinguish between fire smoke and cooking smoke reliably
- Give no visual context or confidence level

This CNN-based system:
- Works **visually** — detects the appearance of fire/smoke in an image
- Can monitor **large areas** from a single camera
- Can be deployed **remotely** without physical installation in every spot
- Provides **confidence scores** indicating certainty of detection
- Can be integrated with **existing CCTV infrastructure** at zero additional hardware cost
- Can be improved continuously by retraining on new data

> In short: smoke detectors tell you *that* something is wrong nearby. This system can tell you *where* and *how certain* — from anywhere a camera can see.

---

 "We developed a Convolutional Neural Network using TensorFlow and Keras to classify images into Fire or No Fire categories. The model was trained on the NASA Space Apps Challenge Fire Detection dataset and deployed as a live web application using Streamlit, achieving 96.57% test accuracy with sub-second inference time."

---

## 👨‍💻 Author

**Hanilasya**
- GitHub: [hanilasya](https://github.com/hanilasya)
