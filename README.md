# 🦠 Malaria Parasite Detection using VGG16

This project utilizes **Transfer Learning** with the **VGG16** architecture to detect malaria-infected blood cells from microscopic images.

## Objective

To develop a binary image classification model that can accurately distinguish:
- **Parasitized** (malaria-infected) cells
- **Uninfected** (healthy) cells

## Dataset

- **Source**: [NIH Malaria Dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)
- **Size**: ~27,000 images (equal class distribution)
- **Type**: Thin blood smear images, cell-level granularity

## Model Overview

- **Base Model**: `VGG16` (pre-trained on ImageNet)
- **Custom Classifier Head**:
  - Flatten
  - Dense (ReLU) + Dropout
  - Dense (Sigmoid) for binary output

- **Frozen Conv Layers**: Initial training done by freezing convolutional base
- **Fine-Tuning**: Top layers of VGG16 unfrozen and retrained with low learning rate

## Tools & Libraries

- TensorFlow / Keras
- scikit-learn
- NumPy
- Matplotlib
- Google Colab

## Training Strategy

- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Regularization**: Dropout, L2 weight decay
- **Data Augmentation**:
  - Random flips
  - Zoom
  - Rotation
- **Callbacks**:
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau

## Evaluation Metrics (on NIH Dataset)

| Metric        | Value  |
| ------------- | ------ |
| Accuracy      | ~96%   |
| F1-Score      | 0.96   |
| Jaccard Index | 0.92   |
| AUC           | ~0.99  |

Confusion matrix showed low false positives and negatives.

## Project Highlights

- Achieved **state-of-the-art performance** using simple fine-tuning and augmentation.
- Demonstrated strong model generalization on clean, labeled biomedical images.
- Efficient training via transfer learning reduced time and computational cost.

## Author

**Lakshya Baranwal**  
B.Tech, Information Technology  
Harcourt Butler Technical University
