# DenseHillNet: Lightweight CNN for Accurate Classification of Natural Images

**Paper:** Saqib et al. (2024), PeerJ Computer Science  
**DOI:** https://doi.org/10.7717/peerj-cs.1995

---

## Team Members

| Name | Roll No | Sections |
|------|---------|----------|
| Yash Dighade | 24AI10049 | 1.  Dataset Preparation , 2. Image Preprocessing & Augmentation |
| Devesh Marya | 24AI10024 | 3. DenseHillNet Architecture, 4. Model Training |
| Nayan Pal | 24AI10039 | 5. Results — Training & Test Metrics, 6. Predicted Values on Test Data |
| Puneet Kumar | 24AI10044 | 7. Confusion Matrix & Classification Report, 8. ROC Curve & Mean Absolute Error |
| Gande Rani | 24AI10008 | 9. Misclassification Analysis, 10. Benchmark Comparison |

---

## Project Overview

This project reproduces the methodology and results from the DenseHillNet paper. The study proposes a DenseNet121-based lightweight CNN architecture for binary classification of natural images into two categories: **Glacier** and **Mountain**. The model achieves **87% accuracy** on a dataset of 3,096 images, outperforming the CNN_OBIA baseline (72%).

---

## Setup & Execution

1. Open `Final_Dl_project.ipynb` on Kaggle or Google Colab
2. Enable GPU runtime (T4 recommended)
3. Download the dataset from Kaggle:  
   [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
4. Run all cells sequentially from top to bottom

---

## Dataset

- **Source:** Intel Image Classification (Kaggle)
- **Classes Used:** Glacier, Mountain (binary classification)
- **Total Images:** 5,994
  - Glacier: 2,957 images
  - Mountain: 3,037 images
- **Train/Test Split:**
  - Training set: 2404 glacier + 2512 mountain
  - Test set: 553 glacier + 525 mountain
- **Preprocessing:**
  - Images resized to 224×224
  - Pixel values rescaled to [0, 1]
  - Augmentation: shear (0.2), zoom (20%), horizontal flip
  - Batch size: 19
  - Class mode: categorical (one-hot encoded)

---

## Methodology

DenseHillNet is built on the **DenseNet121** backbone with custom classification layers added on top:

- **Frozen DenseNet121 base** (pretrained on ImageNet)
- Batch Normalization
- ReLU Activation
- Flatten
- Dense (50 neurons, ReLU)
- Dense (30 neurons, ReLU)
- Dense (2 neurons, Softmax) — output layer

**Total parameters:** 9,552,042  
**Trainable parameters:** 2,512,490  
**Non-trainable parameters:** 7,039,552  

**Training configuration:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 10
- Batch size: 19

---

## Section-wise Breakdown

### Section 1 & 2 — Setup & Dataset Preparation
*(Yash Dighade, 24AI10049)*  
Installs and imports all required libraries (TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn).   
Sets up the dataset directory structure and loads glacier and mountain images using Keras `ImageDataGenerator`. 
Splits data 50/50 into training and test sets.  

### Section 3 & 4 — Preprocessing & Architecture
*(Devesh Marya, 24AI10024)*  
Applies augmentation transforms to the training data including rescaling, shearing, zooming, and horizontal flipping.  
Builds the DenseHillNet model by loading the pretrained DenseNet121 backbone, freezing its weights, and appending custom dense layers.   
Compiles the model with Adam optimizer and categorical crossentropy loss.  

### Section 5 & 6 — Results & Predicted Values
*(Nayan Pal, 24AI10039)*  
Trains the model for 10 epochs and records epoch-by-epoch training and validation loss and accuracy (Table 3).   
Plots the loss and accuracy curves. Runs `model.predict()` on the test set and displays the raw softmax probability vectors for sample glacier and mountain images (Table 4).   
Maps predictions to class labels and compares against ground truth (Table 5).   

### Section 7 & 8 — Confusion Matrix & ROC Curve
*(Puneet Kumar, 24AI10044)*  
Computes and visualizes the full confusion matrix on the test set.    
Generates the classification report with precision, recall, and F1-score for each class. Plots the ROC curve and computes the AUC score and Mean Absolute Error.  

### Section 9 & 10 — Misclassification Analysis & Benchmark
*(Gande Rani, 24AI10008)*  
Identifies and analyzes misclassified images — false glaciers and false mountains.   
Explains why ambiguous terrain (snowy mountains, bare glaciers) causes errors.   
Compares DenseHillNet against the CNN_OBIA baseline from Robson et al. (2020).

---

## Results

### Training Metrics (Table 3)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.4967 | 0.8147 | 0.3525 | 0.8488 |
| 2 | 0.3194 | 0.8729 | 0.3440 | 0.8664 |
| 3 | 0.2843 | 0.8912 | 0.3443 | 0.8609 |
| 4 | 0.2338 | 0.9099 | 0.3308 | 0.8692 |
| 5 | 0.2378 | 0.9085 | 0.3809 | 0.8340 |
| 6 | 0.2162 | 0.9054 | 0.3702 | 0.8701 |
| 7 | 0.2176 | 0.9101 | 0.3269 | **0.8757** |
| 8 | 0.2113 | 0.9121 | 0.3097 | **0.8757** |
| 9 | 0.1928 | 0.9237 | 0.3928 | 0.8525 |
| 10 | 0.1909 | 0.9260 | 0.3807 | 0.8711 |

**Best Validation Accuracy: 87.57% (Epoch 7 & 8)**

### Sample Predictions (Table 4)

| Image | P(Glacier) | P(Mountain) |
|-------|-----------|-------------|
| mountain/20058.jpg | 0.004044 | 0.995956 |
| mountain/20068.jpg | 0.010456 | 0.989544 |
| mountain/20071.jpg | 0.724907 | 0.275093 |
| mountain/20085.jpg | 0.000736 | 0.999264 |
| mountain/20093.jpg | 0.907763 | 0.092237 |
| glacier/20059.jpg | 0.999952 | 0.000048 |
| glacier/20087.jpg | 0.999999 | 0.000001 |
| glacier/20092.jpg | 0.999685 | 0.000315 |
| glacier/20109.jpg | 0.122755 | 0.877245 |
| glacier/20111.jpg | 0.999367 | 0.000633 |

### Predicted vs Actual Classes (Table 5)

| Image | Actual Class | Predicted Class | Decision |
|-------|-------------|-----------------|----------|
| mountain/20058.jpg | Mountain | Mountain | ✅ True Mountain |
| mountain/20068.jpg | Mountain | Mountain | ✅ True Mountain |
| mountain/20071.jpg | Mountain | Glacier | ❌ False Glacier |
| mountain/20085.jpg | Mountain | Mountain | ✅ True Mountain |
| mountain/20093.jpg | Mountain | Glacier | ❌ False Glacier |
| glacier/20059.jpg | Glacier | Glacier | ✅ True Glacier |
| glacier/20087.jpg | Glacier | Glacier | ✅ True Glacier |
| glacier/20092.jpg | Glacier | Glacier | ✅ True Glacier |
| glacier/20109.jpg | Glacier | Mountain | ❌ False Mountain |
| glacier/20111.jpg | Glacier | Glacier | ✅ True Glacier |

### Confusion Matrix

| | Pred Glacier | Pred Mountain |
|---|---|---|
| **Act Glacier** | 459 | 94 |
| **Act Mountain** | 45 | 480 |

### Classification Report (Table 7)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glacier | 0.91 | 0.83 | 0.87 |
| Mountain | 0.84 | 0.91 | 0.87 |
| Macro Avg | 0.87 | 0.87 | 0.87 |
| **Accuracy** | | | **0.87** |

### ROC Curve & Error Metrics

- **AUC:** 0.9396  
- **MAE:** 0.1289

### Misclassification Analysis

- **Total misclassified:** 139 out of 1078
  - False Glacier (Mountains predicted as Glacier): 45
  - False Mountain (Glaciers predicted as Mountain): 94

### Benchmark Comparison (Table 9)

| Model | Accuracy |
|-------|----------|
| CNN_OBIA (Robson et al., 2020) | 72.0% |
| **DenseHillNet (Ours)** | **87.1%** |

**Improvement: +15.1 percentage points**

---

## Repository Structure
```
DenseHillNet-Classification/
│
├── Final_Dl_project.ipynb    ← Main notebook (all 10 sections)
└── README.md
```
---

## References

Saqib SM, Zubair Asghar M, Iqbal M, Al-Rasheed A, Amir Khan M, Ghadi Y, Mazhar T. 2024.   
DenseHillNet: a lightweight CNN for accurate classification of natural images.   
*PeerJ Computer Science* 10:e1995. https://doi.org/10.7717/peerj-cs.1995   
