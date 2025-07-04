# 🧠 Support Vector Machine (SVM) Classifier – Boston Housing Dataset

This project is part of an AI/ML internship (Task 7), where we apply Support Vector Machines (SVM) using Scikit-learn for binary classification on the **Boston Housing dataset**.

---

## 🔍 Objective

- Use SVMs for **linear and non-linear classification**
- Visualize performance using confusion matrix and classification report
- Tune SVM hyperparameters (`C`, `gamma`) using **GridSearchCV**
- Evaluate performance using **cross-validation**

---

## 📁 Dataset

**File:** `BostonHousing.csv`

- `medv`: Median value of owner-occupied homes (target)
- We convert this into a **binary classification problem**:
  - `1`: Expensive (`medv > median`)
  - `0`: Cheap (`medv <= median`)

---

## 🧪 Steps Performed

1. **Load and preprocess the dataset**
2. **Impute missing values**
3. **Scale features using StandardScaler**
4. **Split into train-test sets**
5. **Train SVM with linear and RBF kernel**
6. **Tune hyperparameters using GridSearchCV**
7. **Evaluate using confusion matrix, classification report, and cross-validation**

---

## ✅ Best Hyperparameters (via GridSearchCV)

```json
{
  "C": 10,
  "gamma": 0.1,
  "kernel": "rbf"
}
📊 Results
🎯 Accuracy (Test Set): 86%
🔁 Cross-Validation Mean Accuracy: 81.21%
📌 Confusion Matrix
Predicted Cheap (0)	Predicted Expensive (1)
Actual Cheap	   74   	18
Actual Expensive	4	    56

📃 Classification Report
Class	Precision	Recall	F1-score	Support
0 (Cheap)	0.95	0.80	0.87	92
1 (Expensive)	0.76	0.93	0.84	60

💻 Tech Stack
Python 3.13+

Pandas, NumPy

Scikit-learn

Matplotlib (optional for plotting)
🚀 How to Run

# 1. Clone this repository or download the files
# 2. Install required packages
pip install -r requirements.txt

# 3. Run the Python script
python AL_ML_7.PY

