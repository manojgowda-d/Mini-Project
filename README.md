# Intelligent Computer Vision System for Plant Disease and Pest Detection

An AI-driven agricultural computer vision system that combines **Convolutional Neural Networks (CNN)** for plant disease classification and **Ultralytics YOLOv8** for pest detection and localization.

---

## 📌 Project Overview

The **Intelligent Computer Vision System for Plant Disease and Pest Detection** is designed to assist in the early identification of plant diseases and agricultural pests using deep learning and computer vision.

The system provides two major capabilities:

* 🌿 **Plant Disease Classification** using CNN-based transfer learning.
* 🐛 **Pest Detection and Localization** using YOLOv8 object detection.

The project also includes a preprocessing pipeline, model evaluation utilities, and a **Streamlit web application** for interacting with the trained models.

---

## ✨ Features

### 1. 🌿 Plant Disease Classification

* Multi-class plant disease classification.
* Transfer learning using pretrained CNN architectures.
* Supports architectures such as:

  * ResNet18
  * MobileNetV3
* Predicts the disease class from an uploaded plant/leaf image.
* Provides classification metrics such as:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Generates a confusion matrix and learning curves.

### 2. 🐛 Pest Detection

* Detects agricultural pests using **YOLOv8**.
* Provides bounding-box localization.
* Identifies the location of pests within an image.
* Supports object detection evaluation using:

  * Precision
  * Recall
  * mAP@0.5
  * mAP@0.5:0.95

### 3. 🧹 Dataset Processing

The project contains a dataset preparation pipeline for:

* Dataset inspection
* Corrupted image detection
* Duplicate image detection
* MD5-based duplicate checking
* Data cleaning
* Train/validation/test splitting
* Stratified splitting for disease classification
* YOLO-format dataset preparation

### 4. 📊 Model Evaluation

The system provides evaluation outputs including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Training/validation learning curves
* mAP@0.5
* mAP@0.5:0.95

### 5. 🖥️ Streamlit Web Application

The Streamlit application provides:

* Image upload
* Plant disease prediction
* Pest detection
* Bounding-box visualization
* Model evaluation dashboard
* Dataset inspection information

---

# 🛠️ Technology Stack

| Component              | Technology                   |
| ---------------------- | ---------------------------- |
| Programming Language   | Python                       |
| Deep Learning          | PyTorch                      |
| Disease Classification | CNN / ResNet18 / MobileNetV3 |
| Pest Detection         | YOLOv8                       |
| Computer Vision        | OpenCV                       |
| Web Interface          | Streamlit                    |
| Data Processing        | NumPy, Pandas                |
| Visualization          | Matplotlib                   |
| Model Evaluation       | Scikit-learn                 |
| Configuration          | YAML                         |
| Version Control        | Git & GitHub                 |
| Dataset                | Kaggle                       |

---

# 📁 Project Structure

```text
Mini-Project/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   ├── disease_config.yaml
│   ├── pest_config.yaml
│   └── pest_dataset.yaml
│
├── data/
│   ├── raw/
│   │   ├── disease/
│   │   └── pest/
│   │
│   ├── processed/
│   │
│   ├── disease/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── pest/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       │
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
│
├── models/
│   ├── cnn/
│   └── yolov8/
│
├── src/
│   ├── data/
│   │   ├── inspect_dataset.py
│   │   ├── clean_dataset.py
│   │   └── split_dataset.py
│   │
│   ├── preprocessing/
│   │   └── image_preprocessing.py
│   │
│   ├── disease_detection/
│   │   ├── train_cnn.py
│   │   ├── evaluate_cnn.py
│   │   └── predict_disease.py
│   │
│   ├── pest_detection/
│   │   ├── train_yolov8.py
│   │   ├── evaluate_yolov8.py
│   │   └── predict_pest.py
│   │
│   └── utils/
│       └── metrics.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── Mini-Project Proposal 26-27.docx
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/manojgowda-d/Mini-Project.git
```

Navigate into the project:

```bash
cd Mini-Project
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you are using a GPU, install the appropriate PyTorch version for your CUDA environment.

---

# 📂 Dataset Preparation

The datasets used in this project should be placed inside the `data/raw/` directory.

Recommended structure:

```text
data/
└── raw/
    ├── disease/
    └── pest/
```

The disease dataset should contain images organized according to their classes.

The pest dataset should contain images and corresponding YOLO annotations.

> Dataset files are not included in this repository when they are too large. Download and prepare the datasets separately.

---

# 🔍 Dataset Processing Pipeline

## Step 1: Inspect Dataset

Run:

```bash
python src/data/inspect_dataset.py
```

This performs dataset analysis such as:

* Image count
* Class distribution
* Corrupted image detection
* Duplicate detection
* Image information analysis
* Pest annotation inspection

---

## Step 2: Clean Dataset

Run:

```bash
python src/data/clean_dataset.py
```

This process can:

* Identify corrupted images
* Identify duplicate images
* Remove or isolate invalid files
* Prepare cleaned data for further processing

---

## Step 3: Split Dataset

Run:

```bash
python src/data/split_dataset.py
```

The dataset can be divided into:

```text
70% → Training
15% → Validation
15% → Testing
```

The split is performed before model training to reduce data leakage between training and evaluation sets.

---

# 🧠 Disease Detection Model

The disease detection module uses a CNN-based image classification approach.

## Architecture

The project supports transfer learning using pretrained architectures such as:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Pretrained CNN
     ↓
Feature Extraction
     ↓
Fully Connected Layer
     ↓
Disease Class
```

Example architecture:

```text
ResNet18
   ↓
Pretrained ImageNet Weights
   ↓
Feature Extraction
   ↓
Dropout
   ↓
Linear Classification Layer
   ↓
Disease Prediction
```

---

# 🚀 Train Disease Classification Model

Run:

```bash
python src/disease_detection/train_cnn.py configs/disease_config.yaml
```

The training process includes:

* Image preprocessing
* Data augmentation
* Transfer learning
* Cross-entropy loss
* Adam optimizer
* Learning-rate scheduling
* Early stopping
* Model checkpointing

Example augmentation techniques:

* Random resized crop
* Horizontal flip
* Vertical flip
* Rotation
* Color jitter

---

# 📊 Evaluate Disease Classification Model

Run:

```bash
python src/disease_detection/evaluate_cnn.py configs/disease_config.yaml
```

The evaluation can generate:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Evaluation metrics

Example output files:

```text
models/cnn/
├── best_disease_model.pth
├── class_mapping.json
├── evaluation_metrics.json
└── confusion_matrix.png
```

---

# 🔎 Disease Prediction

To test the disease model on an individual image:

```bash
python src/disease_detection/predict_disease.py <path_to_leaf_image>
```

Example:

```bash
python src/disease_detection/predict_disease.py test_leaf.jpg
```

The model returns the predicted disease class.

---

# 🐛 Pest Detection Using YOLOv8

The pest detection module uses **Ultralytics YOLOv8** for object detection.

The model detects:

* Pest type
* Bounding-box location
* Confidence score

YOLO annotation format:

```text
class_id center_x center_y width height
```

The coordinates are normalized between `0` and `1`.

---

# 🚀 Train YOLOv8 Pest Detection Model

Run:

```bash
python src/pest_detection/train_yolov8.py configs/pest_config.yaml
```

The training process produces a trained YOLOv8 model.

Example output:

```text
models/yolov8/
└── pest_detector/
    └── weights/
        └── best.pt
```

---

# 📊 Evaluate YOLOv8 Model

Run:

```bash
python src/pest_detection/evaluate_yolov8.py configs/pest_config.yaml
```

Evaluation metrics include:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95

---

# 🔎 Pest Detection Prediction

Run:

```bash
python src/pest_detection/predict_pest.py <path_to_pest_image>
```

Example:

```bash
python src/pest_detection/predict_pest.py test_pest.jpg
```

The model produces bounding boxes around detected pests.

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit web application.

Start the application:

```bash
streamlit run app/streamlit_app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Application Features

### 🌿 Disease Detection

Upload a plant or leaf image and the system predicts the corresponding disease class.

### 🐛 Pest Detection

Upload an agricultural image and YOLOv8 detects pests using bounding boxes.

### 📊 Evaluation Dashboard

The dashboard can display:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* mAP metrics
* Learning curves

---

# 🔬 Research Methodology

The proposed system follows the following workflow:

```text
             Dataset Collection
                    │
                    ↓
             Dataset Inspection
                    │
                    ↓
          Data Cleaning & Validation
                    │
                    ↓
             Duplicate Removal
                    │
                    ↓
             Dataset Splitting
              /             \
             /               \
            ↓                 ↓
    Disease Dataset      Pest Dataset
            │                 │
            ↓                 ↓
      CNN Training       YOLOv8 Training
            │                 │
            ↓                 ↓
      Model Evaluation   Model Evaluation
            │                 │
            └────────┬────────┘
                     ↓
              Streamlit App
                     │
                     ↓
          Image-Based Prediction
```

---

# 🧹 Data Leakage Prevention

Data leakage can lead to unrealistically high model performance.

To reduce this risk, the project includes:

* Duplicate image detection
* MD5 hashing
* Dataset cleaning before splitting
* Separate train, validation, and test sets
* Data augmentation applied only to training data
* Independent test-set evaluation

---

# 📈 Evaluation Metrics

## Classification Metrics

### Accuracy

Measures the percentage of correctly classified images.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted positive samples are actually positive.

### Recall

Measures how many actual positive samples are correctly detected.

### F1-Score

The harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## Object Detection Metrics

The pest detection model uses:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95

These metrics evaluate both object classification and bounding-box localization.

---

# 📊 Results

Actual model results should be added here after completing training and evaluation.

Example:

| Model    | Metric       | Result |
| -------- | ------------ | -----: |
| ResNet18 | Accuracy     |    TBD |
| ResNet18 | Precision    |    TBD |
| ResNet18 | Recall       |    TBD |
| ResNet18 | F1-Score     |    TBD |
| YOLOv8   | Precision    |    TBD |
| YOLOv8   | Recall       |    TBD |
| YOLOv8   | mAP@0.5      |    TBD |
| YOLOv8   | mAP@0.5:0.95 |    TBD |

> **Note:** Results should be replaced with the actual experimental results generated by the trained models. No performance values should be claimed without evaluation.

---

# 📋 Configuration

Model configuration files are stored inside:

```text
configs/
```

### Disease Configuration

```text
configs/disease_config.yaml
```

Contains parameters related to:

* Dataset path
* Model architecture
* Number of classes
* Batch size
* Learning rate
* Number of epochs
* Image size
* Training parameters

### Pest Configuration

```text
configs/pest_config.yaml
```

Contains YOLOv8 training parameters such as:

* Model
* Dataset
* Image size
* Batch size
* Epochs
* Learning rate

---

# 📦 Requirements

Main dependencies include:

```text
Python
PyTorch
Torchvision
Ultralytics
OpenCV
NumPy
Pandas
Scikit-learn
Matplotlib
PyYAML
Streamlit
Pillow
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# 🔐 GitHub and Large Files

Large datasets and trained model weights should generally not be committed directly to the repository.

The `.gitignore` file can exclude:

```text
data/raw/
data/processed/
.venv/
__pycache__/
*.pth
*.pt
*.h5
*.onnx
```

This keeps the GitHub repository smaller and easier to clone.

---

# ⚠️ Limitations

The system has several limitations:

1. Prediction performance depends on the quality and diversity of the training dataset.
2. Poor lighting, blur, occlusion, and complex backgrounds can affect detection.
3. The system may not generalize to plant varieties that are not represented in the training data.
4. Pest detection performance depends on the quality of bounding-box annotations.
5. The system is intended as a computer vision research/educational project and should not replace expert agricultural diagnosis.

---

# 🌱 Future Enhancements

Possible future improvements include:

* More crop and disease classes
* Larger and more diverse datasets
* Mobile application deployment
* Real-time camera-based detection
* Lightweight models for edge devices
* Explainable AI using Grad-CAM
* Multilingual farmer interface
* Voice-based interaction
* Integration with weather information
* Crop-specific treatment recommendations
* Cloud-based model deployment

---

# 👨‍💻 Project Team

**Mini Project — 2026–27**

**Domain:** Computer Vision / Artificial Intelligence / Machine Learning

**Project:** Intelligent Computer Vision System for Plant Disease and Pest Detection

---

# ⚠️ Disclaimer

The predictions and recommendations generated by this project are intended for **research and educational purposes**.

The system should not be treated as a substitute for professional agricultural diagnosis. Users should verify disease or pest identification with qualified agricultural experts before making decisions about crop treatment or chemical application.

---

# 📄 License

This project is developed for academic and educational purposes.

Add an appropriate open-source license if the project is intended for public reuse.
