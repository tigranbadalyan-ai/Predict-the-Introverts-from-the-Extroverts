# Predict-the-Introverts-from-the-Extroverts
“Predicting Introverts and Extroverts using behavioral data with Logistic Regression, Random Forest, and XGBoost.”

# 🧠 Personality Predictor: Introvert vs. Extrovert Classification

## 1. Project Overview 🚀

This project involves building a robust machine learning model to classify individuals as either **Introvert** (0) or **Extrovert** (1) based on behavioral and psychological features. This work was developed as part of the **Kaggle Playground Series (Season 5, Episode 7)**, focusing on a critical binary classification task.

The solution emphasizes data cleanliness, advanced feature engineering, and the use of powerful gradient boosting models.

---

## 2. Technical Stack 🛠️

| Category | Libraries | Purpose |
| :--- | :--- | :--- |
| **Language** | Python | Core programming environment. |
| **Data Handling** | Pandas, NumPy | Data manipulation, cleaning, and array operations. |
| **Modeling** | **XGBoost**, Scikit-learn | High-performance gradient boosting and core ML utilities (scaling, CV). |
| **Evaluation** | `roc_auc_score`, `f1_score`, `accuracy_score` | Quantifying model performance on classification metrics. |

---

## 3. Data & Initial Setup 📊

The dataset comprises various features related to social interaction, time usage, and anxiety.

* **Training Data Size:** 18,524 records.
* **Target Variable:** `Personality` (`Extrovert`=1, `Introvert`=0).

### Key Data Insights (EDA)
* **Missing Values:** Handled across several columns, including `Time_spent_Alone`, `Stage_fear`, and `Drained_after_socializing`.
* **Skewness:** The feature **`Time_spent_Alone`** was identified as strongly right-skewed and required logarithmic transformation.

---

## 4. Data Preprocessing & Feature Engineering ✨

The preparation pipeline was crucial for maximizing the signal-to-noise ratio and model performance.

### 4.1. Data Cleaning
1.  **Missing Value Imputation:**
    * **Numerical Features:** Imputed using the **median** strategy.
    * **Categorical Features:** Imputed using the **mode**.
2.  **Encoding:** Binary features and the target variable were converted to **0/1** using Label Encoding.

### 4.2. Feature Engineering (Custom Indices)

Two powerful composite features were created to provide the models with clear, consolidated indicators:

1.  **`Social_Engagement_Score`** (Primary Extroversion Indicator)
    * **Formula:** $\text{Engagement} = (\text{Social\_event\_attendance} + \text{Friends\_circle\_size} + \text{Going\_outside}) - \text{Time\_spent\_Alone\_log}$

2.  **`Public_Avoidance`** (Primary Introversion/Anxiety Indicator)
    * **Formula:** $\text{Avoidance} = \text{Stage\_fear} + (5 - \text{Going\_outside})$

### 4.3. Feature Scaling
* All non-binary numerical features (including the two new composite scores) were scaled using **`StandardScaler`** to center the data ($\mu=0$) and ensure comparable feature weights.

---

## 5. Model Training & Evaluation 🎯

### 5.1. Model Benchmarks (5-Fold Stratified CV)

| Model | Mean CV Accuracy |
| :--- | :--- |
| **Logistic Regression** | **~0.9687** |
| **Random Forest** | **~ 0.9641** |
| **XGBoostClassifier** | **~0.9682** |

### 5.2. Final Model: XGBoost

The **XGBoostClassifier** was selected as the final model due to its superior performance ($\text{CV Accuracy} \approx 0.825$).

* **Final Output File:** `submission_xgboost.csv`

---

## 6. How to Reproduce 💻

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPO_LINK]
    cd [PROJECT_FOLDER]
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
    ```
3.  **Data Files:**
    Place the required data files (`train.csv`, `test.csv`) from the Kaggle competition into the project root directory.
4.  **Execute the Code:**
    Run the analysis script or Jupyter Notebook sections sequentially to perform all steps and generate the final submission file.
