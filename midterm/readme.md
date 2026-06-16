# Cat vs Dog Image Classification

**AI/Machine Learning Midterm Project**
Student: 林小蓮 (111210552) — National Quemoy University, CS Year 3

---

## AI Usage Declaration

This project was assisted by **Claude (claude.ai)** — used for:
- Suggesting improvements to the model (augmentation, dropout)
- README writing assistance

Initial code and architecture were written by the student. Claude was used to suggest improvements between versions.

---

## Project Overview

A Cat vs Dog image classifier built using Convolutional Neural Networks (CNN) with Python and TensorFlow/Keras. Given an image, the model predicts whether it contains a cat or a dog.

---

## Dataset

- **Source:** [Kaggle — Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Size:** ~25,000 labeled images
- **Split:** 80% train / 20% validation

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Main language |
| TensorFlow / Keras | Model building & training |
| NumPy | Data handling |
| Matplotlib | Visualization |
| Google Colab | Training environment (free GPU) |

---

## Model Architecture

```
Input (150x150x3)
-> Conv2D + ReLU + MaxPooling  (x3)
-> Flatten
-> Dense (512) + ReLU + Dropout
-> Dense (1) + Sigmoid
```

---

## Improvement Process

| Version | Changes | Val Accuracy |
|---------|---------|----------|
| v1 | Basic CNN, written by student | ~70% |
| v2 | Added data augmentation (suggested by AI) | ~82% |
| v3 | Added Dropout to reduce overfitting (suggested by AI) | ~85% |

---

## How to Run

```bash
pip install tensorflow numpy matplotlib
jupyter notebook cat_vs_dog.ipynb
```

Or open directly in Google Colab (recommended).

---

## Results

- Final validation accuracy: ~85%
- Model successfully distinguishes cats from dogs on unseen images
- Sample predictions shown in notebook

---

## File Structure

```
cat-vs-dog/
├── cat_vs_dog.ipynb
├── train.py
├── README.md
└── sample_images/
```

---

## References

- [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- [TensorFlow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
