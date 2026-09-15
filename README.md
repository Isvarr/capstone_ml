# capstone_ml
Machine learning capstone project for course 23CSE301
# Financial Risk Score Prediction

### 23CSE301 — Machine Learning Capstone Project

A complete end-to-end **Machine Learning Regression Pipeline** for predicting the financial **Risk Score** of loan applicants using applicant demographic, employment, credit, income, debt, and loan-related information.

This project is developed as part of the **23CSE301 Machine Learning Capstone Project** and focuses specifically on the **Regression Track**.

---

## 📌 Project Overview

Financial institutions need reliable methods to assess the financial risk associated with loan applicants. A numerical risk score can help quantify an applicant's potential financial instability or likelihood of loan default.

In this project, machine learning regression models are trained to predict the continuous **`RiskScore`** using financial and applicant-related attributes.

The project follows a complete machine learning workflow:

```text
Raw Dataset
     ↓
Data Loading & Audit
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Train-Test Split
     ↓
Regression Model Training
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Best Model Selection
```

---

## 🎯 Problem Statement

> **Predict the financial risk score (`RiskScore`) of a loan applicant based on their demographic, employment, credit history, financial, and loan-related characteristics.**

The problem is formulated as a **supervised regression problem** because the target variable is a continuous numerical value.

### Target Variable

```text
RiskScore
```

The dataset also contains `LoanApproved`, which represents a binary loan approval outcome. However, **this project uses `RiskScore` exclusively as the regression target**.

---

## 📊 Dataset

### Source

The dataset is obtained from Kaggle:

[Financial Risk for Loan Approval — Kaggle](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?utm_source=chatgpt.com)

### Dataset Characteristics

| Property          | Description                      |
| ----------------- | -------------------------------- |
| Dataset           | Financial Risk for Loan Approval |
| Source            | Kaggle                           |
| Type              | Synthetic financial dataset      |
| Records           | 20,000                           |
| Columns           | 36                               |
| Regression Target | `RiskScore`                      |
| Dataset File      | `Loan.csv`                       |
| Regression Task   | Risk Score Prediction            |

The dataset contains demographic information, employment characteristics, income, credit history, debt-related information, assets, liabilities, loan characteristics, interest rates, and other financial indicators.

> **Note:** The dataset is synthetic. Therefore, model performance should be interpreted as performance on the generated dataset rather than as evidence of real-world lending performance.

---

## 🧾 Dataset Features

The dataset contains the following attributes:

### Applicant Information

* `ApplicationDate`
* `Age`
* `MaritalStatus`
* `NumberOfDependents`
* `EducationLevel`
* `EmploymentStatus`
* `Experience`
* `JobTenure`

### Financial Information

* `AnnualIncome`
* `MonthlyIncome`
* `SavingsAccountBalance`
* `CheckingAccountBalance`
* `TotalAssets`
* `TotalLiabilities`
* `NetWorth`

### Credit Information

* `CreditScore`
* `CreditCardUtilizationRate`
* `NumberOfOpenCreditLines`
* `NumberOfCreditInquiries`
* `DebtToIncomeRatio`
* `TotalDebtToIncomeRatio`
* `BankruptcyHistory`
* `PreviousLoanDefaults`
* `PaymentHistory`
* `LengthOfCreditHistory`

### Loan Information

* `LoanAmount`
* `LoanDuration`
* `LoanPurpose`
* `BaseInterestRate`
* `InterestRate`
* `MonthlyLoanPayment`

### Other Outcome Variables

* `LoanApproved`
* `RiskScore`

For this regression project:

```text
Target → RiskScore
```

`LoanApproved` is not used as the regression target.

---

# 🔬 Machine Learning Methodology

## 1. Data Loading & Audit

The dataset is first loaded using Pandas and inspected for:

* Dataset dimensions
* Data types
* Missing values
* Duplicate records
* Numerical and categorical variables
* Target distribution
* Potential outliers

---

## 2. Exploratory Data Analysis

The EDA stage investigates the characteristics and relationships present in the dataset.

### Planned Visualizations

* Feature distribution plots
* Target (`RiskScore`) distribution
* Correlation heatmap
* Boxplots for numerical variables
* Feature vs. target scatter plots
* Financial feature relationships
* Outlier analysis

Each major visualization is accompanied by an interpretation explaining the observed pattern and its relevance to the regression problem.

---

## 3. Data Preprocessing

The preprocessing pipeline includes:

### Missing Values

Missing values are identified and handled using an appropriate strategy based on the feature type and distribution.

### Duplicate Records

Duplicate observations are checked and appropriately handled.

### Outlier Analysis

Numerical features are analyzed for extreme observations using statistical and visualization-based techniques.

### Categorical Encoding

Categorical variables are transformed into numerical representations suitable for machine learning algorithms.

### Feature Scaling

Scaling is applied where required, particularly for algorithms sensitive to feature magnitude such as:

* SVR
* KNN
* Linear models
* Polynomial Regression

To prevent data leakage, preprocessing transformations are fitted using the training data and then applied to the test data.

---

# 🛠️ Regression Algorithms

The project implements all **10 regression algorithms required by the 23CSE301 capstone guidelines**.

| #  | Algorithm                      | Purpose                                |
| -- | ------------------------------ | -------------------------------------- |
| 1  | Linear Regression              | Baseline regression model              |
| 2  | Ridge Regression               | L2 regularization                      |
| 3  | Lasso Regression               | L1 regularization and feature sparsity |
| 4  | ElasticNet Regression          | Combination of L1 + L2 regularization  |
| 5  | Polynomial Regression          | Capture nonlinear relationships        |
| 6  | Decision Tree Regressor        | Nonlinear rule-based regression        |
| 7  | Random Forest Regressor        | Ensemble regression                    |
| 8  | Gradient Boosting Regressor    | Sequential boosting-based regression   |
| 9  | Support Vector Regressor (SVR) | Margin-based nonlinear regression      |
| 10 | K-Nearest Neighbors Regressor  | Distance-based regression              |

All models are trained using the **same preprocessed dataset and the same held-out test set** to ensure a fair comparison.

---

# 📏 Evaluation Metrics

The models are evaluated using:

### R² Score

Measures how much of the variance in `RiskScore` is explained by the model.

**Higher is better.**

### Root Mean Squared Error (RMSE)

Measures the average magnitude of prediction errors while giving larger errors greater weight.

**Lower is better.**

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual risk scores.

**Lower is better.**

### 5-Fold Cross-Validated R²

The two best-performing models are further evaluated using **5-fold cross-validation** to assess their generalization performance.

The capstone guidelines specifically require R², RMSE and MAE for the model comparison, with 5-fold CV R² for the two best models.

---

# ⚙️ Hyperparameter Tuning

Hyperparameter optimization is performed on at least two regression models using:

```text
GridSearchCV
```

or

```text
RandomizedSearchCV
```

Examples of parameters considered include:

### Ridge

```text
alpha
```

### Lasso

```text
alpha
```

### ElasticNet

```text
alpha
l1_ratio
```

### Decision Tree

```text
max_depth
min_samples_split
min_samples_leaf
```

### Random Forest

```text
n_estimators
max_depth
min_samples_split
```

### Gradient Boosting

```text
n_estimators
learning_rate
max_depth
```

### SVR

```text
C
kernel
gamma
```

### KNN

```text
n_neighbors
weights
```

The tuned models are compared against their corresponding baseline versions to determine whether hyperparameter optimization improves performance.

---

# 📊 Model Comparison

The final regression comparison will contain:

| Model                 | R² | RMSE | MAE | CV R² |
| --------------------- | -: | ---: | --: | ----: |
| Linear Regression     |  — |    — |   — |     — |
| Ridge Regression      |  — |    — |   — |     — |
| Lasso Regression      |  — |    — |   — |     — |
| ElasticNet            |  — |    — |   — |     — |
| Polynomial Regression |  — |    — |   — |     — |
| Decision Tree         |  — |    — |   — |     — |
| Random Forest         |  — |    — |   — |     — |
| Gradient Boosting     |  — |    — |   — |     — |
| SVR                   |  — |    — |   — |     — |
| KNN                   |  — |    — |   — |     — |

> Results will be populated after training and evaluating all models.

Models will primarily be ranked according to **R²**, while RMSE and MAE will be used to provide additional insight into prediction error.

---

# 📈 Required Visualizations

The project includes the following visualizations:

### Exploratory Analysis

* Feature distributions
* RiskScore distribution
* Correlation heatmap
* Feature-target scatter plots
* Outlier analysis

### Model Evaluation

* Actual vs. Predicted RiskScore
* Residual plot
* Model performance comparison
* Feature importance for tree-based models

The capstone rubric specifically requires a **residual plot**, **predicted-vs-actual plot for the best model**, and **feature importance for at least one tree-based model**.

---

# 🧠 Best Model Selection

The final model will be selected based on a combination of:

* High test-set R²
* Low RMSE
* Low MAE
* Strong 5-fold cross-validation R²
* Generalization performance
* Model complexity
* Interpretability

Rather than selecting a model solely based on one metric, the final selection will consider both predictive performance and consistency across validation results.

---

# 📁 Repository Structure

The repository follows the structure recommended by the 23CSE301 capstone guidelines:

```text
financial-risk-regression/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── Loan.csv
│
├── notebooks/
│   └── regression.ipynb
│
├── models/
│   └── best_model.pkl
│
└── app/
    └── ...
```

The capstone guidelines recommend a root `README.md`, `requirements.txt`, dataset directory, notebooks, optional saved models, and optional application/deployment code.

---

# 💻 Technologies Used

* **Python 3**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

---

# 🚀 Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd financial-risk-regression
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/regression.ipynb
```

Run the notebook **from top to bottom**.

The complete pipeline includes:

```text
Data Loading
      ↓
Data Audit
      ↓
EDA
      ↓
Cleaning
      ↓
Feature Engineering
      ↓
Encoding
      ↓
Train/Test Split
      ↓
Scaling
      ↓
10 Regression Models
      ↓
Hyperparameter Tuning
      ↓
Evaluation
      ↓
5-Fold Cross Validation
      ↓
Model Comparison
      ↓
Best Model
```

---

# 🔒 Reproducibility

A fixed random seed is used wherever applicable:

```python
random_state = 42
```

The same train/test split is maintained across all regression algorithms so that model comparisons remain fair.

Preprocessing steps such as scaling and encoding are fitted only on the training data to avoid **data leakage**, as required by the capstone guidelines.

---

# 📌 Expected Outcomes

The project aims to:

1. Understand the factors associated with financial risk scores.
2. Develop multiple regression models for `RiskScore` prediction.
3. Compare linear, nonlinear, regularized, ensemble, distance-based, and kernel-based regression techniques.
4. Identify the best-performing regression model.
5. Evaluate model generalization using cross-validation.
6. Analyze important features influencing risk-score predictions.
7. Demonstrate a complete and reproducible machine learning regression pipeline.

---

# ⚠️ Dataset & Project Limitations

* The dataset is **synthetic**, so its patterns may not fully represent real-world borrowers.
* Model performance on this dataset does not guarantee equivalent performance on real financial data.
* Financial risk is influenced by many external factors that may not be represented in the dataset.
* Correlations observed in the dataset should not automatically be interpreted as causal relationships.
* The model is developed for academic and experimental purposes and **should not be used for real-world loan approval decisions without additional validation and governance**.

---

# 🎓 Academic Context

This project is developed for:

**Course:** 23CSE301 — Machine Learning
**Program:** B.Tech. Computer Science and Engineering
**Academic Year:** 2026–27
**Track:** Regression

The capstone requires an end-to-end ML pipeline covering data loading, EDA, preprocessing, feature engineering, model training, comparison, hyperparameter tuning, and visualization.

The regression track requires implementation and comparison of **10 specified regression algorithms** using a consistent evaluation methodology.

---

# 👥 Team

| Name                      | Roll Number      |
| ------------------------- | ---------------- |
| Sidambarisvar Balamurugan | CB.SC.U4CSE24751 |
| Niranjan Reddy            | CB.SC.U4CSE24735 |
| Mithesh G S               | CB.SC.U4CSE24715 |

---

# 📚 References

* **Dataset:** [Financial Risk for Loan Approval — Kaggle](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?utm_source=chatgpt.com)
* **Course:** 23CSE301 Machine Learning — Capstone Project Guidelines
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## 🤖 AI Assistance

Generative AI tools may be used for code scaffolding and development assistance. Any AI-assisted components are reviewed, tested, and understood by the project team. Analysis, interpretation, feature-engineering decisions, and final conclusions are independently evaluated by the team.

This project follows the academic-integrity requirements specified in the 23CSE301 capstone guidelines.
