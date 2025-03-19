# Diabetes Prediction using SVM

## 📌 Project Overview
This project predicts whether a person is diabetic or not using **Support Vector Machine (SVM)**. The dataset consists of medical features that help in classifying individuals as diabetic or non-diabetic.

## 📂 Dataset
- The dataset contains medical diagnostic features such as glucose level, blood pressure, BMI, etc.
- The target variable indicates whether a person is **diabetic (1) or not diabetic (0).**
- Source: https://www.dropbox.com/scl/fi/0uiujtei423te1q4kvrny/diabetes.csv?rlkey=20xvytca6xbio4vsowi2hdj8e&e=1&dl=0

## 🛠️ Technologies & Libraries Used
- Python
- Pandas (for data manipulation)
- NumPy (for numerical computations)
- Scikit-learn (for ML modeling)
- Seaborn & Matplotlib (for data visualization)

## 🚀 How to Run the Project
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/jiggsnoway/diabetes-prediction-svm.git
cd diabetes-prediction-svm
```

### 2️⃣ Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3️⃣ Run the Model
```python
python diabetes_prediction.py
```

## 🔬 Model Training & Evaluation
### 1️⃣ Data Preprocessing
- Missing values are handled.
- Data is standardized using **StandardScaler**.

### 2️⃣ Splitting Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

### 3️⃣ Training the Model
```python
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, Y_train)
```

### 4️⃣ Evaluating the Model
```python
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
```

## 📊 Confusion Matrix Visualization
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## 📜 Results
- Accuracy: **77%**
- Precision, Recall, F1-score are evaluated.

## 🤖 Future Improvements
- Try different kernels for SVM (RBF, Polynomial).
- Experiment with hyperparameter tuning.
- Use other ML models for comparison.

---
Made with ❤️ by Jigyashman Hazarika

