# 🔭 Gamma Ray vs Hadron Classification using Machine Learning and Deep Learning

This project is a complete machine learning pipeline for classifying gamma-ray and hadron signals using real-world astronomical data from the 

# 🎥 Demo Video
https://github.com/user-attachments/assets/ec789127-9e1f-4395-8a20-08b6209b105d


**MAGIC Telescope**. It compares multiple classification algorithms including traditional ML models and a neural network built using TensorFlow.

**Model training repository**: https://github.com/pulindu-seniya-silva/magic-dataset-Machine-Learning/blob/main/Magic.ipynb


<img src="images/1.png" width="800" alt="image 1" />
<img src="images/2.png" width="800" alt="image 1" />
---

## 📁 Dataset

- **Name**: `magic04.data`
- **Source**: UCI Machine Learning Repository
- **Features**:
  - `fLength`, `fWidth`, `fSize`, `fConc`, `fConcl`, `fAsym`, `fM3Long`, `FM3Trans`, `fAlpha`, `fDist`
  - `class`: `"g"` (gamma ray) or `"h"` (hadron) → converted to `1` and `0`
- **Model training repository**: https://github.com/pulindu-seniya-silva/magic-dataset-Machine-Learning/blob/main/Magic.ipynb

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
