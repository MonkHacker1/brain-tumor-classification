# Brain Tumor Classification with Deep Learning 🧠

A complete machine learning pipeline for the automated classification of four classes of brain conditions (**Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**) from human MRI images. This project implements a custom **OpenCV-based preprocessing** module to standardize and clean the medical images before they are fed into a **ResNet50** convolutional neural network.

---

## 🎯 Importance of the Subject

**Early detection and classification of brain tumors** is a critical domain in medical imaging. The ability to accurately and quickly classify tumors in terms of grade and type is vital for selecting the most convenient and life-saving treatment method for patients. Deep learning approaches, particularly those using **Convolutional Neural Networks (CNNs)**, are providing impactful solutions to improve health diagnosis in this field.

---

## 🚀 Key Features

* **Custom Preprocessing:** Implements a contour-based image cropping function to find and isolate the **Brain Region of Interest (ROI)**, effectively removing surrounding margins and noise artifacts. This step is crucial for improving model accuracy.
* **Dataset Handling:** Handles images of varying sizes by resizing them to a uniform dimension (e.g., 256x256) after preprocessing.
* **Transfer Learning:** Utilizes the state-of-the-art **ResNet50** pre-trained model for robust and efficient image classification.
* **End-to-End Pipeline:** Includes scripts for data extraction (`image_extractor.py`), cleaning (`preprocessing.py`), and the final model training notebook (`Brain_Tumor_Classification.ipynb`).

---

## 📂 Dataset Details

The dataset used in this project is a combination of three sources: **Figshare**, **SARTAJ**, and **Br35H**.

| Statistic | Details |
| :--- | :--- |
| **Source** | [Kaggle: Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| **Total Images** | 7022 MRI images |
| **Classes** | **4 Classes:** glioma, meningioma, pituitary, no tumor |
| **Split** | ~22% for testing, ~78% for training |
| **Note** | Images in the original dataset have different sizes, necessitating the preprocessing step to remove extra margins and standardize dimensions. |

***Note on Data Extraction:*** The original `image_extractor.py` was designed to handle `.mat` files, common in medical datasets like Figshare. If you are using the provided Kaggle dataset link, ensure your initial data is structured appropriately or modify `image_extractor.py` to handle the specific image file structure of the downloaded data.

---

## 🛠️ Basic Requirements

The project was developed and tested using the following package versions:

| Package Name | Version |
| :--- | :--- |
| **Python** | 3.7.12 |
| **tensorflow** | 2.6.0 |
| **keras** | 2.6.0 |
| **keras-preprocessing** | 1.1.2 |
| **matplotlib** | 3.0.2 |
| **opencv** | 4.1.2 |
| **scikit-learn** | 0.22.2 |
| **h5py** | (For .mat file handling) |
| **Pillow** (PIL) | (For image manipulation) |

### Installation

```bash
# Recommended installation for the project environment
pip install numpy tqdm opencv-python imutils h5py Pillow \
    tensorflow==2.6.0 keras==2.6.0 keras-preprocessing==1.1.2 \
    matplotlib==3.0.2 scikit-learn==0.22.2
```
# ⚙️ How to Run
### 1. Data Structure

Ensure your data is organized as follows:

```bash
brain-tumor-classification/
├── dataset/
│   ├── Training/
│   └── Testing/
├── cleaned/
├── preprocessing.py
└── Brain_Tumor_Classification.ipynb
```
### 2. Data Pre-processing (preprocessing.py)
```bash
This script performs the custom OpenCV-based cropping and resizing.
# Input: Images from 'dataset/Training' and 'dataset/Testing'
# Output: Cleaned images saved to 'cleaned/Training' and 'cleaned/Testing'
python preprocessing.py
```
### 3. Model Training (Brain_Tumor_Classification.ipynb)
```bash
Use a Jupyter environment to load the cleaned data and train the ResNet50 classification model
# Open the Jupyter environment
jupyter notebook
# Navigate to and run all cells in 'Brain_Tumor_Classification.ipynb'
```
### 4. Results and Evaluation

The custom image preprocessing pipeline combined with the fine-tuned ResNet50 model achieved outstanding results on the test set.

Final Classification Metrics:

The model achieved an overall **accuracy of 0.99 (99%)** on the test set, demonstrating excellent performance across all four classes. The detailed classification report is below (where class 0-3 correspond to the tumor types/No Tumor):
***Classification Report***

| Class         | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|-----------|----------|
| 0              | 0.98      | 0.99   | 0.98      | 294      |
| 1              | 0.98      | 0.97   | 0.98      | 303      |
| 2              | 1.00      | 1.00   | 1.00      | 403      |
| 3              | 0.99      | 0.99   | 0.99      | 294      |
| **Macro Avg**  | **0.99**  | **0.99** | **0.99** | **1294** |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** | **1294** |

