# 🔭 Gamma Ray vs Hadron Classification using Machine Learning and Deep Learning

This project is a complete machine learning pipeline for classifying gamma-ray and hadron signals using real-world astronomical data from the **MAGIC Telescope**. It compares multiple classification algorithms including traditional ML models and a neural network built using TensorFlow.

---

## 📁 Dataset

- **Name**: `magic04.data`
- **Source**: UCI Machine Learning Repository
- **Features**:
  - `fLength`, `fWidth`, `fSize`, `fConc`, `fConcl`, `fAsym`, `fM3Long`, `FM3Trans`, `fAlpha`, `fDist`
  - `class`: `"g"` (gamma ray) or `"h"` (hadron) → converted to `1` and `0`

---

## 📌 Project Structure

- 🔍 **Data Preprocessing**: Cleaning, feature scaling, class label encoding
- 📊 **Visualization**: Histograms for feature distributions by class
- ⚖️ **Oversampling**: Handled class imbalance using `RandomOverSampler`
- 🔀 **Data Split**: Train (60%), Validation (20%), Test (20%)
- 🤖 **ML Models**:
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
- 🧠 **Neural Network** (TensorFlow):
  - 2 hidden layers + dropout
  - Hyperparameter tuning: learning rate, dropout, batch size
  - Validation-based model selection
- 📈 **Evaluation**: `classification_report()` on test set (Precision, Recall, F1-score, Accuracy)

---

## 🧪 Installation & Requirements

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow
