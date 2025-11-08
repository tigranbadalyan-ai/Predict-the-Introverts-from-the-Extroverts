# Predict-the-Introverts-from-the-Extroverts
“Predicting Introverts and Extroverts using behavioral data with Logistic Regression, Random Forest, and XGBoost.”

# 🧠 Personality Predictor: Introvert vs. Extrovert Classification

## 1️⃣ Project Overview

This project focuses on building a machine learning model to classify individuals as **Introvert (0)** or **Extrovert (1)** based on behavioral, social, and psychological features.  

The work was developed as part of the **Kaggle Playground Series (Season 5, Episode 7)** and represents a binary classification challenge.

> The main goal: create an accurate and interpretable model to distinguish introverts from extroverts.

---

## 2️⃣ Technical Stack

| Category | Libraries | Purpose |
| :--- | :--- | :--- |
| **Language** | Python | Main programming environment |
| **Data Handling** | Pandas, NumPy | Data loading, cleaning, and manipulation |
| **Modeling** | XGBoost, Random Forest, Logistic Regression, Scikit-learn | Machine learning models, scaling, cross-validation |
| **Visualization** | Matplotlib, Seaborn | Plotting and feature analysis |
| **Evaluation** | `accuracy_score`, `f1_score`, `roc_auc_score` | Classification performance metrics |

---

## 3️⃣ Data & Initial Insights

* **Train Data:** 18,524 records  
* **Test Data:** provided by Kaggle competition  
* **Target Variable:** `Personality` (`Extrovert`=1, `Introvert`=0)  

### Key EDA steps:

* Checked **missing values** and imputed them (median for numerical, mode for categorical)  
* Analyzed **feature distributions**, identified strong right skew in `Time_spent_Alone` → applied log transformation  
* Visualized class balance and feature distributions  

---

## 4️⃣ Data Preprocessing & Feature Engineering

### 4.1. Data Cleaning

1. Filling missing values:
   * **Numerical:** median  
   * **Categorical:** mode
2. Encoding categorical variables and target as **0/1** (`LabelEncoding`)  
   * `Stage_fear`, `Drained_after_socializing`, `Personality`  

### 4.2. Feature Engineering

**Social_Engagement_Score** — primary extroversion indicator:

\[
\text{Social\_Engagement\_Score} = (\text{Social\_event\_attendance} + \text{Friends\_circle\_size} + \text{Going\_outside}) - \log(1+\text{Time\_spent\_Alone})
\]

**Public_Avoidance** — primary introversion/anxiety indicator:

\[
\text{Public\_Avoidance} = \text{Stage\_fear} + (5 - \text{Going\_outside})
\]

### 4.3. Feature Scaling

* StandardScaler applied to numeric and composite features (`Social_Engagement_Score`, `Public_Avoidance`)  
* Binary features remained unchanged  

---

## 5️⃣ Model Training & Evaluation

### 5.1. Logistic Regression

* **5-Fold Stratified CV Accuracy:** ~0.965  
* **Train Accuracy:** ~0.963  
* **Train F1 Score:** ~0.966 
* **Train ROC-AUC:** ~0.963

### 5.2. Random Forest

* **5-Fold Stratified CV Accuracy:** ~0.972 
* **Train Accuracy:** ~0.966
* **Train F1 Score:** ~0.977 
* **Train ROC-AUC:** ~0.975

### 5.3. XGBoostClassifier (Final Model)

* Selected as the **final model** due to highest cross-validation accuracy and ability to handle composite features  
* **5-Fold Stratified CV Accuracy:** ~0.976 
* **Train Accuracy:** ~0.967
* **Train F1 Score:** ~0.977 
* **Train ROC-AUC:** ~0.965

*Confusion Matrix on training data:*  

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("XGBoost Confusion Matrix — Train Data")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

