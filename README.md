 Cats vs Dogs Image Classification using Deep Learning

Overview

This project implements a Convolutional Neural Network (CNN) from scratch to classify images of cats and dogs. The model was developed using TensorFlow and Keras in Google Colab and trained on the Cats vs Dogs dataset from Kaggle.

The objective of this project was not only to build an image classification model but also to study the impact of different optimization techniques such as Batch Normalization, Dropout, Data Augmentation, Early Stopping, Learning Rate Scheduling, and Optimizer Comparison on model performance.

---
Dataset

Dataset: Cats vs Dogs Dataset (Kaggle)

- Training Images: 20,000
- Validation Images: 5,000
- Classes:
  - Cats
  - Dogs

The dataset was downloaded directly using KaggleHub and loaded using TensorFlow's `image_dataset_from_directory()` utility. :contentReference[oaicite:0]{index=0}

---

Technologies Used

- Python
- TensorFlow 2.20
- Keras
- NumPy
- Matplotlib
- Google Colab
- KaggleHub

---

Model Architecture

The CNN model was built entirely from scratch without using any pre-trained network.

Architecture:

- Input Layer (128 × 128 × 3)
- Convolution Layer (32 Filters)
- Batch Normalization
- Max Pooling
- Dropout

- Convolution Layer (64 Filters)
- Batch Normalization
- Max Pooling
- Dropout

- Convolution Layer (128 Filters)
- Batch Normalization
- Max Pooling
- Dropout

- Global Average Pooling
- Dense Layer (256 Units)
- Batch Normalization
- Dropout
- Output Layer (Sigmoid)

The complete model architecture includes three convolution blocks followed by a lightweight classifier head using Global Average Pooling. :contentReference[oaicite:1]{index=1}

---
 Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 128 × 128 |
| Batch Size | 32 |
| Maximum Epochs | 30 |
| Learning Rate | 0.001 |
| Dropout Rate | 0.5 |
| Optimizer | Adam |

---

Techniques Applied

The following techniques were implemented to improve model performance:

- Data Augmentation
  - Random Horizontal Flip
  - Random Rotation
  - Random Zoom

- Batch Normalization

- Dropout Regularization

- Early Stopping

- ReduceLROnPlateau Learning Rate Scheduler

- Model Checkpointing

- Optimizer Comparison (Adam, SGD and RMSProp)

---

Results

The model was trained using multiple optimizers to compare their performance.

| Optimizer | Validation Accuracy |
|-----------|--------------------|
| Adam | **85.80%** |
| RMSProp | **82.66%** |
| SGD | **64.52%** |

Among all tested optimizers, Adam achieved the highest validation accuracy for this implementation. :contentReference[oaicite:2]{index=2}

---

 Performance Analysis

During training, the project includes:

- Training and Validation Accuracy Curves
- Training and Validation Loss Curves
- Learning Rate Scheduling
- Overfitting Analysis
- Confidence-based Predictions
- Optimizer Performance Comparison

These evaluations helped analyze model behavior and identify the most suitable optimizer for the dataset.

---

 Project Structure

```
Cats-vs-Dogs-Classification/
│
├── Cats_vs_Dogs_Classification.ipynb
├── README.md
├── requirements.txt
├── best_catsdogs_model.h5
├── accuracy.png
├── loss.png
├── optimizer_comparison.png
└── predictions.png
```

---

Future Improvements

Possible future enhancements include:

- Transfer Learning using EfficientNet or MobileNet
- Hyperparameter Tuning
- Model Deployment with Streamlit
- Real-time Image Prediction
- TensorFlow Lite Conversion for Mobile Devices

---

Author

Nimra Maqbool

BS Computer Science

University of Management and Technology (UMT), Lahore
