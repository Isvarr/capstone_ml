# capstone_ml

**23CSE301 — Machine Learning Capstone Project**

A complete, end-to-end machine learning project covering **all three capstone tracks** on three different datasets:

| Track | Task | Dataset | Notebook |
|-------|------|---------|----------|
| 📈 **Regression** | Predict a continuous financial `RiskScore` for loan applicants | Financial Risk for Loan Approval (`Loan.csv`) | `credit_risk_regression.ipynb` |
| 🧠 **Classification** | Predict whether a patient will have a stroke (`stroke` = 0/1) | Stroke Prediction (`healthcare-dataset-stroke-data.csv`) | `classification_stroke_final.ipynb` |
| 🧩 **Clustering** | Discover natural groups from eating habits and physical condition (unsupervised) | Obesity Levels (`ObesityDataSet_raw_and_data_sinthetic.csv`) | `clustering.ipynb` |

Every track follows the same disciplined workflow: **data audit → EDA → cleaning → feature engineering → encoding/scaling → modelling → evaluation → model selection**, with a fixed `random_state = 42` and leakage-safe preprocessing (all transformers are fitted on training data only).

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Datasets](#-datasets)
3. [Track 1 — Regression](#-track-1--regression-loan-risk-score-prediction)
4. [Track 2 — Classification](#-track-2--classification-stroke-prediction)
5. [Track 3 — Clustering](#-track-3--clustering-obesity-levels)
6. [Repository Structure](#-repository-structure)
7. [Technologies Used](#-technologies-used)
8. [Installation](#-installation)
9. [Running the Project](#-running-the-project)
10. [Reproducibility](#-reproducibility)
11. [Limitations](#-limitations)
12. [Academic Context](#-academic-context)
13. [Team](#-team)
14. [References](#-references)
15. [AI Assistance](#-ai-assistance)

---

## 📌 Project Overview

The capstone requires an end-to-end ML pipeline for each track. This repository contains one notebook per track.

```
Raw Dataset → Data Loading & Audit → EDA → Cleaning → Feature Engineering
            → Encoding → Scaling → Train/Test Split (supervised tracks only)
            → Model Training → Hyperparameter Tuning → Evaluation
            → Model Comparison → Final Model Selection
```

| | Regression | Classification | Clustering |
|---|---|---|---|
| **Learning type** | Supervised | Supervised | Unsupervised |
| **Target** | `RiskScore` (continuous) | `stroke` (binary, 4.87% positive) | None (`NObeyesdad` used *only* for post-hoc validation) |
| **Algorithms** | 10 regressors | 10 classifiers | K-Means, Agglomerative (hierarchical) |
| **Key metrics** | R², RMSE, MAE, 5-fold CV R² | Precision, Recall, F1, ROC-AUC, 5-fold CV ROC-AUC | Silhouette, Davies-Bouldin, Calinski-Harabasz (+ ARI/NMI post-hoc) |
| **Best result** | Tuned Gradient Boosting (XGBoost), R² = **0.9008** | Tuned Logistic Regression, ROC-AUC = **0.8387** | K-Means (k = 4), Silhouette = **0.155** |

---

## 📊 Datasets

| Track | Dataset | Rows × Columns | Source | Notes |
|-------|---------|----------------|--------|-------|
| Regression | `Loan.csv` | 20,000 × 36 | [Kaggle — Financial Risk for Loan Approval](https://www.kaggle.com/) | Synthetic. No missing values or duplicates. |
| Classification | `healthcare-dataset-stroke-data.csv` | 5,110 × 12 | Kaggle — Stroke Prediction Dataset | 201 missing `bmi` values; severe class imbalance (249 strokes, 4.87%). |
| Clustering | `ObesityDataSet_raw_and_data_sinthetic.csv` | 2,111 × 17 | UCI / Kaggle — Obesity Levels | ~75% SMOTE-generated synthetic samples; 24 duplicate rows. |

> **Note:** The regression and clustering datasets are (partly) synthetic. Results reflect performance on the generated data and should not be taken as evidence of real-world performance.

---

## 📈 Track 1 — Regression: Loan Risk Score Prediction

**Notebook:** `credit_risk_regression.ipynb`

### Problem
Predict the continuous `RiskScore` (range ≈ 29–84, mean ≈ 50.8) of a loan applicant from demographic, employment, credit, income, debt and loan attributes.

`LoanApproved` is **deliberately excluded**: it is a binary decision downstream of the risk score (correlation ≈ −0.77 with `RiskScore`), so using it as a feature would be direct target leakage.

### Preprocessing & Feature Engineering
- **Cleaning:** defensive median/mode imputation, duplicate removal, IQR-based outlier detection on continuous columns only (binary flags excluded), outliers **winsorized** rather than dropped.
- **Engineered features:** `LoanToIncomeRatio` (loan size relative to repayment capacity), `ApplicationYear`, `ApplicationMonth` (extracted from `ApplicationDate`).
- **Encoding:** `EducationLevel` ordinal-encoded; `EmploymentStatus`, `MaritalStatus`, `HomeOwnershipStatus`, `LoanPurpose` one-hot encoded (`drop_first=True`) → **44 features**.
- **Split & scaling:** 80:20 split stratified on a 10-bin quantile discretisation of `RiskScore`; `StandardScaler` fitted on the training set only (binary dummies not scaled).

### Algorithms (all 10 required)

| # | Algorithm | Notes |
|---|-----------|-------|
| 1 | Linear Regression | Baseline; coefficients interpreted |
| 2 | Ridge Regression | L2 regularisation |
| 3 | Lasso Regression | L1; zeroed out 23 / 44 coefficients |
| 4 | ElasticNet | L1 + L2 |
| 5 | Polynomial Regression | Degrees 1–3 on the top-8 correlated features; degree 3 best |
| 6 | Decision Tree Regressor | Non-linear, rule-based |
| 7 | Random Forest Regressor | Bagged ensemble |
| 8 | Gradient Boosting Regressor | Implemented with **XGBoost** |
| 9 | Support Vector Regressor | RBF kernel on scaled features |
| 10 | K-Nearest Neighbors Regressor | Distance-based; k scanned on scaled data |

### Hyperparameter Tuning
Six models were tuned (the requirement is at least two) using `GridSearchCV` / `RandomizedSearchCV`. For the slowest models (Random Forest, SVR) the search ran on a 4,000-row training subsample and the winning configuration was refit on the full training set.

| Model | Search | Best parameters | Test R² (baseline → tuned) |
|-------|--------|-----------------|----------------------------|
| Ridge | `GridSearchCV` | `alpha=1` | 0.8173 → 0.8173 |
| Decision Tree | `GridSearchCV` | `max_depth=None, min_samples_leaf=10` | 0.6905 → 0.7549 |
| Random Forest | `RandomizedSearchCV` | `n_estimators=100, min_samples_leaf=2, max_depth=None` | 0.8603 → 0.8614 |
| Gradient Boosting (XGBoost) | `RandomizedSearchCV` | `n_estimators=200, max_depth=7, learning_rate=0.05` | 0.8889 → 0.9008 |
| SVR | `GridSearchCV` | `C=10, kernel='rbf'` | 0.8159 → 0.8592 |
| KNN | `GridSearchCV` | `n_neighbors=31, weights='distance'` | 0.5650 → 0.5674 |

### Final Model Comparison (held-out test set, tuned where applicable)

| Rank | Model | R² | RMSE | MAE | 5-Fold CV R² |
|------|-------|----|------|-----|--------------|
| 1 | **Gradient Boosting (XGBoost, tuned)** | **0.9008** | **2.438** | **1.502** | **0.9034 ± 0.0033** |
| 2 | Random Forest (tuned) | 0.8614 | 2.882 | 1.772 | 0.8652 ± 0.0027 |
| 3 | SVR (tuned) | 0.8592 | 2.905 | 2.127 | — |
| 4 | Ridge Regression (tuned) | 0.8173 | 3.309 | 2.599 | — |
| 5 | Linear Regression | 0.8173 | 3.309 | 2.600 | — |
| 6 | Lasso Regression | 0.8018 | 3.447 | 2.700 | — |
| 7 | Decision Tree (tuned) | 0.7549 | 3.834 | 2.372 | — |
| 8 | ElasticNet | 0.7546 | 3.835 | 2.954 | — |
| 9 | Polynomial Regression (deg 3) | 0.6481 | 4.593 | 3.492 | — |
| 10 | KNN (tuned) | 0.5674 | 5.092 | 3.803 | — |

5-fold CV R² is reported for the two best-performing models, as required, and is computed on the training set only.

### Key Findings
- **Best model:** tuned Gradient Boosting (XGBoost) — R² = 0.9008, and its CV R² (0.9034) closely matches the test R², so the result is not an artefact of one split.
- The boosted/ensemble tree models clearly beat the linear models, indicating **non-linear structure** in the data.
- **Most influential drivers** (agreed by both linear coefficients and tree importances): `BankruptcyHistory`, `PreviousLoanDefaults`, income, and debt-to-income ratios.
- KNN performs worst — distance-based methods struggle in a 44-dimensional space.
- Residuals are symmetric around zero with no funnel or curve, and predicted-vs-actual points hug the diagonal.

---

## 🧠 Track 2 — Classification: Stroke Prediction

**Notebook:** `classification_stroke_final.ipynb`

### Problem
Predict whether a patient has a stroke (`stroke` = 0/1) from demographic and health attributes. Only **249 of 5,110 patients (4.87%)** had a stroke, so the project is built around the **class-imbalance problem**: accuracy and weighted F1 are misleading, and **ROC-AUC, precision and recall** drive every decision.

### Preprocessing & Feature Engineering
- **Cleaning:** dropped the single `gender = 'Other'` row; `bmi` outliers **winsorized** at the IQR fences (dropping rows would lose rare positive cases); missing `bmi` imputed with the median *inside the pipeline* (training data only).
- **Split:** stratified 80:20 on `stroke` (test set: 1,022 rows, 50 strokes).
- **Pipeline:** scikit-learn `ColumnTransformer` + `Pipeline` — median imputation and `StandardScaler` for numerics, ordinal encoding for binary features, one-hot encoding for `work_type` and `smoking_status`. The fit-on-train-only rule is enforced automatically.
- **Engineered feature:** `age_glucose_risk = age × avg_glucose_level`, justified by EDA showing strokes concentrated where age *and* glucose are both elevated. Its correlation with `stroke` (≈0.27) exceeds that of either parent feature.
- **Imbalance handling:** `class_weight='balanced'` where supported; balanced sample weights for AdaBoost, Gradient Boosting and MLP.

### Algorithms (all 10 required)

| Part | # | Algorithm |
|------|---|-----------|
| A | 1 | Logistic Regression |
| A | 2 | K-Nearest Neighbors |
| A | 3 | Naive Bayes (Gaussian) |
| A | 4 | Decision Tree |
| A | 5 | Support Vector Machine (RBF) |
| B | 6 | Random Forest |
| B | 7 | AdaBoost |
| B | 8 | Gradient Boosting |
| B | 9 | Bagging (Decision Tree base) |
| B | 10 | MLP (Neural Network) |

Part B models were tuned with **stratified 5-fold `GridSearchCV`** (scoring = ROC-AUC, training set only). Tuning improved the test ROC-AUC of every Part B model — e.g. MLP 0.757 → 0.841, Random Forest 0.781 → 0.826, Gradient Boosting 0.810 → 0.837.

### Consolidated Comparison (held-out test set)

| # | Model | Part | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC | CV ROC-AUC (train) |
|---|-------|------|----------|-----------|--------|---------------|---------|--------------------|
| 1 | **Logistic Regression** | A | 0.7339 | 0.1325 | 0.80 | 0.8093 | 0.8378 | **0.8385** |
| 2 | K-Nearest Neighbors | A | 0.9472 | 0.0000 | 0.00 | 0.9253 | 0.6017 | 0.6424 |
| 3 | Naive Bayes (Gaussian) | A | 0.3992 | 0.0740 | 0.98 | 0.5194 | 0.8018 | 0.8128 |
| 4 | Decision Tree | A | 0.6869 | 0.1006 | 0.68 | 0.7759 | 0.7988 | 0.7570 |
| 5 | SVM (SVC) | A | 0.7387 | 0.1220 | 0.70 | 0.8125 | 0.7935 | 0.7991 |
| 6 | Random Forest (tuned) | B | 0.9217 | 0.2321 | 0.26 | 0.9238 | 0.8263 | 0.8305 |
| 7 | AdaBoost (tuned) | B | 0.6937 | 0.1210 | 0.84 | 0.7807 | 0.8383 | 0.8384 |
| 8 | Gradient Boosting (tuned) | B | 0.6908 | 0.1200 | 0.84 | 0.7785 | 0.8365 | 0.8279 |
| 9 | Bagging (tuned) | B | 0.7055 | 0.1254 | 0.84 | 0.7892 | 0.8390 | 0.8276 |
| 10 | MLP (tuned) | B | 0.7329 | 0.1320 | 0.80 | 0.8086 | 0.8410 | 0.8380 |

> ⚠️ **Why accuracy and weighted F1 mislead here:** KNN has the *highest* accuracy (94.7%) and weighted F1 (0.925) yet a recall of **0.00** — it never detects a stroke. Models are therefore ranked by **cross-validated ROC-AUC**, a threshold-independent metric that is selected on training data only.

### Final Model Selection
The top three models by CV ROC-AUC — **Logistic Regression (0.8385), AdaBoost (0.8384) and MLP (0.8380)** — are statistically tied. Logistic Regression was chosen because it is the simplest, most interpretable (odds ratios), trains instantly and gives well-behaved probabilities.

**Final model:** Logistic Regression (`C=0.01`, `penalty='l1'`, `class_weight='balanced'`) with a decision threshold of **≈ 0.64**, selected by maximising F2 (recall weighted twice as heavily as precision) on out-of-fold training predictions — appropriate for a screening task where a missed stroke is costlier than a false alarm.

| Version (test set) | Accuracy | Precision | Recall | F1 (stroke) | ROC-AUC |
|--------------------|----------|-----------|--------|-------------|---------|
| Default LR (t = 0.50) | 0.7339 | 0.1325 | 0.80 | 0.2273 | 0.8378 |
| Tuned LR (t = 0.50) | 0.6957 | 0.1217 | 0.84 | 0.2127 | 0.8387 |
| **Tuned LR + F2 threshold (t = 0.64) — FINAL** | **0.8249** | **0.1823** | **0.74** | **0.2925** | **0.8387** |
| Tuned LR + F1 threshold (t = 0.75) | 0.8963 | 0.2500 | 0.56 | 0.3457 | 0.8387 |

### Key Findings
- **Age is the dominant driver** of stroke risk (top tree split, largest logistic coefficient, highest Random Forest importance — by both impurity and permutation), followed by the engineered `age_glucose_risk` feature.
- **Extra model complexity buys almost nothing:** six models cluster at ROC-AUC 0.826–0.841, so a simple linear model plus one interaction term captures most of the available signal.
- Precision is low for every model (≈0.07–0.23) because strokes are rare — any screening tool on this data will produce many false alarms.

---

## 🧩 Track 3 — Clustering: Obesity Levels

**Notebook:** `clustering.ipynb`

### Problem
Discover natural groupings of individuals from lifestyle and body measurements **without using labels during fitting**. The ground-truth column `NObeyesdad` (7 obesity classes) is held aside and used **only afterwards** to interpret and validate the clusters.

### Preprocessing
- 24 exact duplicates removed → **2,087 rows**.
- `NObeyesdad` separated into `y_true` and never enters the feature matrix.
- `StandardScaler` on 8 numeric features (mandatory for distance-based methods); one-hot encoding for categoricals → **26 features**.
- No train/test split (unsupervised — all rows are clustered).

### Algorithms
| # | Algorithm | Configuration |
|---|-----------|---------------|
| 1 | **K-Means** | `k = 4`, `n_init=10` |
| 2 | **Agglomerative Hierarchical** | Ward linkage, `k = 4` |

**Choosing *k*:** the elbow curve has no sharp elbow, but the largest bend (max second difference of inertia) is at **k = 4**, which is also the local silhouette peak for small *k*. The Ward dendrogram has its largest gap in merge heights at the same place (4 clusters), so both algorithms use `k = 4` and are directly comparable.

**Linkage comparison:** single and average linkage give deceptively good silhouette/DBI scores because they split off 1–few outliers (*chaining*); complete linkage also produces a tiny cluster. **Ward** is the only linkage giving four reasonably sized clusters (smallest = 293 points), so it was selected.

### Results

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | ARI (post-hoc) | NMI (post-hoc) |
|-----------|--------------|------------------|---------------------|----------------|----------------|
| **K-Means (k = 4)** | **0.1552** | **1.9922** | **281.12** | 0.1795 | 0.2474 |
| Agglomerative (Ward, k = 4) | 0.1389 | 2.0647 | 255.75 | 0.1879 | 0.2657 |

K-Means wins on all three internal metrics; Ward is marginally better on agreement with the true labels — the two methods are effectively comparable.

### K-Means Cluster Profiles

| Cluster | Size | Description |
|---------|------|-------------|
| 1 | 742 | Young (mean age 20.6), lean (mean BMI 23.6), most physically active — mostly Insufficient / Normal weight |
| 2 | 703 | Severe obesity (mean weight 113.7 kg, mean BMI 37.3) — dominated by Obesity Type III |
| 3 | 340 | Older (mean age 35.8), overweight/obese (BMI 30.3), lowest screen time |
| 0 | 302 | Separated by a very low number of main meals (`NCP` ≈ 1.2 vs ≈ 3) — an eating-pattern group, not a weight group |

### Visualisations
Elbow + silhouette + Davies-Bouldin curves, Ward dendrogram (and a 4-linkage comparison), PCA 2-D plots for every algorithm (with K-Means centroids), t-SNE plots for both algorithms, and cluster-vs-true-label heatmaps.

### Key Findings
- **Weak cluster separation** (silhouette ≈ 0.15): the data behaves like a continuum rather than distinct groups. This is expected with 26 mixed features and ~75% synthetic (SMOTE-interpolated) samples. A low silhouette here is a property of the data, not a bug.
- Agreement with the 7 true classes is modest (ARI ≈ 0.18, NMI ≈ 0.25–0.27): the classes are defined by BMI cut-offs, whereas the clusters mix body-size and behavioural information.
- **Possible extensions:** dropping near-constant features (e.g. `SMOKE`), clustering in PCA space, or density-based / Gaussian-mixture models.

---

## 📁 Repository Structure

```
capstone_ml/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── Loan.csv                                       # Regression
│   ├── healthcare-dataset-stroke-data.csv             # Classification
│   └── ObesityDataSet_raw_and_data_sinthetic.csv      # Clustering
│
├── notebooks/
│   ├── credit_risk_regression.ipynb
│   ├── classification_stroke_final.ipynb
│   └── clustering.ipynb
│
├── processed/                 # Auto-generated by the regression notebook (Section B)
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── models/                    # Optional: saved best models
│   └── ...
│
└── app/                       # Optional: deployment / demo code
    └── ...
```

> The notebooks load their CSV with a relative path (e.g. `pd.read_csv('Loan.csv')`). Either run each notebook from the folder containing its dataset, or update the `DATA_PATH` / `read_csv` line near the top to point to `../data/...`.

---

## 💻 Technologies Used

- Python 3
- Pandas, NumPy
- Scikit-learn
- XGBoost (regression Gradient Boosting)
- SciPy (hierarchical clustering, dendrograms, cophenetic correlation)
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🚀 Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd capstone_ml
```

Create and activate a virtual environment:

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```
numpy
pandas
scipy
scikit-learn>=1.7   # the classification notebook passes sample_weight to MLPClassifier
xgboost
matplotlib
seaborn
jupyter
```

---

## ▶️ Running the Project

Launch Jupyter:

```bash
jupyter notebook
```

Then open the notebook for the track you want and run it **top to bottom**:

| Track | Notebook | Notes |
|-------|----------|-------|
| Regression | `credit_risk_regression.ipynb` | Section B writes `processed/*.csv`; Section C reloads them, so C can be re-run independently once those files exist. |
| Classification | `classification_stroke_final.ipynb` | Part A (5 algorithms) and Part B (5 algorithms + tuning + final selection) are in one notebook. |
| Clustering | `clustering.ipynb` | t-SNE and the K-Means sweep for *k* are the slowest cells. |

---

## 🔒 Reproducibility

- `random_state = 42` is used everywhere it applies (splits, models, KFold/StratifiedKFold, t-SNE, KMeans).
- Within each supervised track, **the same train/test split and preprocessing are used for every algorithm**, so comparisons are fair.
- Imputers, scalers and encoders are **fitted on training data only**, then applied to the test data (via `Pipeline`/`ColumnTransformer` for classification, and explicit train-only fitting for regression), to avoid data leakage.
- Hyperparameters are tuned with cross-validation on the **training set only**; the test set is used only for final reporting.

---

## ⚠️ Limitations

**General**
- Model performance on these datasets does not guarantee equivalent performance on real-world data.
- Correlations and feature importances should not be read as causal relationships.
- This is an academic project and **must not be used for real lending decisions, medical diagnosis or health screening** without extensive additional validation.

**Regression**
- The dataset is synthetic, so its patterns may not represent real borrowers.
- Random Forest and SVR hyperparameter searches ran on a 4,000-row subsample for tractability.

**Classification**
- The test set contains only 50 stroke cases, so each patient moves recall by 0.02; small differences between models are not statistically meaningful.
- The imputer/scaler were fitted on the whole training set before cross-validation (not refitted per fold), which can make CV scores very slightly optimistic. Held-out test scores are unaffected.
- The top three models are tied on ROC-AUC; Logistic Regression was chosen for simplicity and interpretability, not a significant performance gap.
- The MLP hit `max_iter=500` in some grid configurations (convergence warnings).

**Clustering**
- About three quarters of the data is synthetic (SMOTE-generated), which blurs cluster boundaries.
- Silhouette scores ≈ 0.15 indicate weak, overlapping clusters; the choice of `k = 4` is a parsimonious judgement call rather than a clear optimum.
- PCA's first two components explain only ≈ 36% of the variance, so 2-D plots are a lossy view; t-SNE distances between far-apart blobs should not be over-interpreted.

---

## 🎓 Academic Context

- **Course:** 23CSE301 — Machine Learning
- **Program:** B.Tech. Computer Science and Engineering
- **Academic Year:** 2026–27
- **Tracks covered:** Regression, Classification, Clustering

The capstone requires an end-to-end ML pipeline covering data loading, EDA, preprocessing, feature engineering, model training, comparison, hyperparameter tuning and visualisation, with a consistent evaluation methodology across algorithms.

---

## 👥 Team

| Name | Roll Number |
|------|-------------|
| Sidambarisvar Balamurugan | CB.SC.U4CSE24751 |
| Niranjan Reddy | CB.SC.U4CSE24735 |
| Mithesh G S | CB.SC.U4CSE24715 |

---

## 📚 References

- Dataset: *Financial Risk for Loan Approval* — Kaggle
- Dataset: *Stroke Prediction Dataset* — Kaggle (`healthcare-dataset-stroke-data.csv`)
- Dataset: *Estimation of Obesity Levels Based on Eating Habits and Physical Condition* (`ObesityDataSet_raw_and_data_sinthetic.csv`)
- Course: 23CSE301 Machine Learning — Capstone Project Guidelines
- Libraries: Pandas, NumPy, Scikit-learn, XGBoost, SciPy, Matplotlib, Seaborn

---

## 🤖 AI Assistance

Generative AI tools may be used for code scaffolding and development assistance. Any AI-assisted components are reviewed, tested and understood by the project team. Analysis, interpretation, feature-engineering decisions and final conclusions are independently evaluated by the team.

This project follows the academic-integrity requirements specified in the 23CSE301 capstone guidelines.
