# 🏡 California Housing Price Prediction (Regression)

This project predicts median house prices in California districts using the **California Housing dataset** from scikit-learn.

The goal of this lab is to practice:

- Regression modeling  
- Feature scaling  
- Hyperparameter tuning  
- Comparing linear vs tree-based models  

---

## 📂 Project Structure

```
california-housing-ml/
│
├── main.py               # End-to-end ML pipeline
├── requirements.txt      # Dependencies
└── README.md
```

---

## 📊 Dataset

We use the built-in dataset:

```
from sklearn.datasets import fetch_california_housing
```

**Features include:**

- Median income  
- House age  
- Average rooms  
- Population  
- Latitude, longitude  
- …and more  

**Target variable:**

```
MedHouseVal  → median house value (in $100,000 units)
```

---

## 🤖 Models Used

### **1. Ridge Regression (Linear Model)**  
- Uses StandardScaler  
- Hyperparameter tuning with GridSearchCV  
- Evaluated using **RMSE, MAE, R²**

---

### **2. Random Forest Regressor (Tree Model)**  
- Non-linear model  
- No scaling required  
- Compared against Ridge  

---

## ▶️ How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the project:

```
python main.py
```

---

## 📈 Results (Example Output)

```
=== Ridge Regression Results ===
RMSE: 0.74
MAE : 0.53
R²  : 0.57

=== Random Forest Regression Results ===
RMSE: 0.50
MAE : 0.32
R²  : 0.80
```

Random Forest generally performs better because it captures **non-linear patterns** in the housing data.

---

## ✅ Summary

- Ridge (linear) learns a smooth relationship  
- Random Forest captures complex, non-linear interactions  
- Feature scaling is required only for linear models  
- Tree models work well out-of-the-box  

---

