# German Credit Risk Prediction

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://statistic---german-credit-project-ep15.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A complete machine-learning project for **German Credit Risk Classification**, designed to predict whether a loan applicant should be classified as **Good Credit** or **Bad Credit**.

The project focuses on **cost-sensitive classification** rather than accuracy alone. Missing a genuinely bad-credit applicant is treated as substantially more costly than incorrectly flagging a good-credit applicant.

> **Live application:**  
> https://statistic---german-credit-project-ep15.streamlit.app/

> **Repository:**  
> https://github.com/tnamgia76-debug/Statistic---German-Credit-Project

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Business Objective](#2-business-objective)
- [3. Dataset Structure](#3-dataset-structure)
- [4. Methodology](#4-methodology)
- [5. Preprocessing](#5-preprocessing)
- [6. Model Search Space](#6-model-search-space)
- [7. Threshold Optimization](#7-threshold-optimization)
- [8. Feature Selection Strategy](#8-feature-selection-strategy)
- [9. Nested Cross-Validation](#9-nested-cross-validation)
- [10. Final Model](#10-final-model)
- [11. Benchmark Results](#11-benchmark-results)
- [12. Competition Submission](#12-competition-submission)
- [13. Streamlit Application](#13-streamlit-application)
- [14. Repository Structure](#14-repository-structure)
- [15. Installation](#15-installation)
- [16. Running the Project](#16-running-the-project)
- [17. Deployment](#17-deployment)
- [18. Reproducibility and Leakage Control](#18-reproducibility-and-leakage-control)
- [19. Limitations](#19-limitations)
- [20. Future Improvements](#20-future-improvements)

---

# 1. Project Overview

The goal of this project is to build a reproducible binary classification pipeline for credit-risk prediction.

Each applicant is assigned an estimated probability of belonging to the **Bad Credit** class:

```text
bad_credit = 0  -> Good Credit
bad_credit = 1  -> Bad Credit
```

The final decision is produced by applying a classification threshold to the predicted probability:

```text
P(Bad) >= threshold  -> Bad Credit
P(Bad) < threshold   -> Good Credit
```

The project includes:

- Exploratory Data Analysis
- Reproducible preprocessing pipelines
- Multiple machine-learning algorithms
- Hyperparameter search
- Cost-sensitive threshold optimization
- Nested Cross-Validation
- EDA-based feature-set comparison
- Historical benchmark evaluation
- Competition prediction
- Streamlit deployment
- Single-applicant and batch credit-risk prediction

---

# 2. Business Objective

The primary evaluation objective is the asymmetric business cost:

\[
\text{Cost} = 5 \times FN + FP
\]

where:

- **FN — False Negative:** an actual Bad applicant predicted as Good
- **FP — False Positive:** an actual Good applicant predicted as Bad

A False Negative is therefore considered **five times more costly** than a False Positive.

This means the project does **not** optimize accuracy alone.

A model may intentionally accept additional False Positives if doing so substantially reduces the number of Bad applicants incorrectly classified as Good.

### Confusion-matrix interpretation

| Actual | Predicted Good | Predicted Bad |
|---|---:|---:|
| Good | TN | FP |
| Bad | FN | TP |

Bad Credit (`1`) is treated as the positive class.

---

# 3. Dataset Structure

The original labeled dataset contains **800 observations**.

A fixed stratified split is used:

| Dataset | Good | Bad | Total | Purpose |
|---|---:|---:|---:|---|
| Development | 480 | 160 | 640 | Model development, Nested CV and final selection |
| Historical Benchmark | 120 | 40 | 160 | Post-selection diagnostic evaluation |
| Competition | Unknown | Unknown | 200 | Unlabeled final predictions |

The class distribution in both labeled subsets is approximately:

```text
Good: 75%
Bad : 25%
```

The 200 competition rows do not contain public target labels.

---

# 4. Methodology

The full modelling workflow is:

```text
800 labeled observations
        |
        +------------------------------+
        |                              |
640 Development                160 Historical Benchmark
        |                              |
        |                              |
Nested Cross-Validation               |
        |                              |
Evaluate selection procedure          |
        |                              |
Final search on 640                   |
        |                              |
Freeze final rule --------------------+
        |
Fit on 640
        |
Evaluate on benchmark
        |
NO further tuning
        |
Refit frozen rule on all 800
        |
Predict 200 competition rows
        |
Export model artifacts
        |
Deploy Streamlit application
```

The key principle is:

> **Inner CV selects. Outer CV evaluates.**

The historical benchmark is not used to choose a model, threshold, feature set or random seed.

---

# 5. Preprocessing

The project uses a `scikit-learn` Pipeline so that preprocessing is fitted only on the relevant training fold.

## Numerical features

The original numerical variables are:

```text
duration
amount
age
```

Pipeline:

```text
Median Imputation
        |
StandardScaler
```

## Categorical features

The remaining input variables are handled as categorical variables even when represented by integer codes.

Pipeline:

```text
Most-Frequent Imputation
        |
OneHotEncoder(
    handle_unknown="ignore"
)
```

This preprocessing design helps prevent data leakage because all learned preprocessing statistics are fitted only on training data.

---

# 6. Model Search Space

The project evaluates **9 model families** and **37 predefined configurations**.

| Model | Hyperparameters | Configurations |
|---|---|---:|
| Dummy | Baseline | 1 |
| Logistic Regression | `C ∈ {0.1, 1, 10}`, `class_weight ∈ {None, balanced}` | 6 |
| Gaussian Naive Bayes | `var_smoothing ∈ {1e-9, 1e-3}` | 2 |
| K-Nearest Neighbors | `n_neighbors ∈ {5, 15, 31}`, `weights ∈ {uniform, distance}` | 6 |
| Decision Tree | `max_depth ∈ {3, 6}`, `min_samples_leaf ∈ {5, 15}` | 4 |
| Random Forest | `max_depth ∈ {5, None}`, `min_samples_leaf ∈ {3, 10}` | 4 |
| Extra Trees | `max_depth ∈ {5, None}`, `min_samples_leaf ∈ {3, 10}` | 4 |
| RBF SVM | `C ∈ {0.1, 1, 10}`, `gamma ∈ {scale, 0.01}` | 6 |
| Gradient Boosting | `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {1, 2}` | 4 |

Total:

```text
37 model configurations
```

---

# 7. Threshold Optimization

The project does not automatically use the conventional threshold of `0.50`.

For each candidate model configuration:

1. Generate Out-of-Fold probabilities.
2. Test candidate thresholds.
3. Convert probability into Good/Bad predictions.
4. Compute:

\[
5 \times FN + FP
\]

5. Select the threshold with the lowest cost.

The threshold search used in the final notebook is:

```python
np.arange(0.01, 1.00, 0.01)
```

Therefore, threshold selection is treated as part of model selection.

---

# 8. Feature Selection Strategy

Four predeclared EDA-based feature strategies are compared:

| Strategy | Number of Features | Dropped Features |
|---|---:|---|
| `Full_20` | 20 | None |
| `Drop_2` | 18 | `telephone`, `people_liable` |
| `Drop_4` | 16 | `telephone`, `people_liable`, `other_debtors`, `job` |
| `Drop_6` | 14 | `telephone`, `people_liable`, `other_debtors`, `job`, `housing`, `property` |

The feature set itself is treated as a model-selection choice.

The Inner CV therefore selects a complete prediction rule consisting of:

```text
Feature Set
+
Model Family
+
Hyperparameters
+
Classification Threshold
```

### Nested-CV comparison

The saved project run produced:

```text
Full-20 baseline Nested-CV cost      : 391
Feature-selection Nested-CV cost     : 393
```

This means feature-set selection did **not** improve the Outer-CV cost in this run.

The two procedures performed similarly, while the final full-development search was still allowed to select a simpler reduced-feature rule.

---

# 9. Nested Cross-Validation

The development set contains 640 observations.

## Outer CV

A fixed **5-fold Stratified Cross-Validation** structure is used.

Each iteration contains approximately:

```text
Outer Training   = 512 rows
Outer Validation = 128 rows
```

The Outer Validation set is kept untouched while the complete prediction rule is selected.

## Inner CV

Each 512-row Outer Training set is evaluated using **3-fold Stratified Cross-Validation**.

For each candidate:

```text
3 Inner folds
      |
OOF probabilities for all 512 rows
      |
Threshold optimization
      |
Inner business cost
```

The candidate with the lowest Inner-CV cost is selected.

It is then:

1. Rebuilt from scratch
2. Fitted on all 512 Outer Training observations
3. Evaluated on the untouched 128-row Outer Validation set

This process is repeated across all five Outer folds.

The combined Outer predictions estimate the performance of the **selection procedure**, rather than the performance of one fixed model.

---

# 10. Final Model

After Nested Cross-Validation is completed, the same selection rule is applied to all 640 development observations.

The final selected rule is:

| Component | Selected Value |
|---|---|
| Feature strategy | `Drop_6` |
| Number of features | 14 |
| Model | Logistic Regression |
| `C` | `0.1` |
| `class_weight` | `balanced` |
| Classification threshold | `0.39` |
| Final-search selection cost | `318` |

The 14 selected features are:

```text
status
duration
credit_history
purpose
amount
savings
employment_duration
installment_rate
personal_status_sex
present_residence
age
other_installment_plans
number_credits
foreign_worker
```

Important:

> The final-search cost of `318` is a **selection criterion**, not an unbiased test-performance estimate.

The unbiased procedure-level evaluation comes from the Outer CV.

---

# 11. Benchmark Results

Once the final rule was frozen, it was fitted on the 640 development observations and evaluated on the 160-row historical benchmark.

## Confusion Matrix

|  | Predicted Good | Predicted Bad |
|---|---:|---:|
| Actual Good | 68 | 52 |
| Actual Bad | 10 | 30 |

Therefore:

```text
TN = 68
FP = 52
FN = 10
TP = 30
```

## Metrics

| Metric | Result |
|---|---:|
| Business Cost | **102** |
| Accuracy | 0.6125 |
| Precision — Bad | 0.3659 |
| Recall — Bad | **0.7500** |
| ROC-AUC | 0.7296 |
| Average Precision | 0.5055 |
| Brier Score | 0.1969 |

The model identified **75% of the actual Bad-credit cases**.

The relatively high number of False Positives reflects the asymmetric project objective: reducing False Negatives is prioritized because each FN carries five times the cost of an FP.

No model, feature set or threshold is changed after this benchmark evaluation.

---

# 12. Competition Submission

After the benchmark diagnostic step, the development and benchmark datasets are combined:

```text
640 Development
+
160 Historical Benchmark
=
800 labeled observations
```

The frozen final rule is then refitted on all 800 labeled observations.

Predictions are generated for the 200 unlabeled competition rows.

## Label conversion

The internal model target is:

```text
bad_credit = 0 -> Good
bad_credit = 1 -> Bad
```

The competition submission uses the original `kredit` encoding:

```text
kredit = 1 -> Good
kredit = 0 -> Bad
```

Therefore:

```python
competition_kredit = 1 - competition_pred
```

The final submission schema is:

```text
Id,kredit
```

### Kaggle leaderboard

The submitted model received:

```text
Public score  : 0.70
Private score : 0.60
```

These leaderboard scores are reported separately from the project's business-cost objective.

The modelling pipeline was designed to minimize:

\[
5 \times FN + FP
\]

rather than to directly maximize the external competition score.

---

# 13. Streamlit Application

A Streamlit application is provided to demonstrate the trained model interactively.

## Live App

https://statistic---german-credit-project-ep15.streamlit.app/

## Application Features

### Single Applicant

A user can enter one applicant's information and receive:

- Good / Bad credit-risk classification
- Predicted probability of Bad Credit
- Final decision threshold
- Visual risk indicator

The deployed app applies the same final threshold:

```text
0.39
```

### Batch Prediction

Users can upload a CSV file containing multiple applicants.

The application:

1. Checks required features
2. Generates `P(Bad)` for every row
3. Applies the frozen threshold
4. Returns Good / Bad predictions
5. Allows prediction results to be downloaded

### Model Information

The application displays:

- Final model family
- Number of selected features
- Decision threshold
- Business-cost definition
- Model hyperparameters
- Selected input variables

## Deployment Artifacts

The application loads the already-fitted model rather than retraining it.

```text
artifacts/
├── final_model.joblib
└── model_metadata.joblib
```

`final_model.joblib` contains the complete fitted preprocessing + model pipeline.

`model_metadata.joblib` contains deployment information such as:

```text
model_name
params
features
threshold
numeric_features
categorical_features
category_values
numeric_defaults
```

---

# 14. Repository Structure

```text
Statistic---German-Credit-Project/
│
├── main_v2.ipynb
│   └── Main modelling, evaluation and export notebook
│
├── app.py
│   └── Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│
├── artifacts/
│   ├── final_model.joblib
│   └── model_metadata.joblib
│
├── models/
│   ├── dummy.py
│   ├── logistic.py
│   ├── gaussian_nb.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── extra_trees.py
│   ├── rbf_svm.py
│   └── gradient_boosting.py
│
└── data_splits_colab/
    └── data_split/
        ├── development_640.csv
        ├── historical_benchmark_160.csv
        └── competition_200.csv
```

The exact repository contents may include additional project-support files.

---

# 15. Installation

## 15.1 Clone the repository

```bash
git clone https://github.com/tnamgia76-debug/Statistic---German-Credit-Project.git
cd Statistic---German-Credit-Project
```

## 15.2 Create a virtual environment

### Windows

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 15.3 Install dependencies

```bash
python -m pip install -r requirements.txt
```

Tested package versions:

```text
Python          3.13
Streamlit       1.64.0
pandas          2.3.3
NumPy           2.2.3
scikit-learn    1.7.2
joblib          1.5.2
```

---

# 16. Running the Project

## Run the Streamlit app locally

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Run the modelling notebook

Open:

```text
main_v2.ipynb
```

Run the notebook from the repository root.

The notebook uses portable project paths:

```python
PROJECT_ROOT = Path.cwd().resolve()
```

This avoids machine-specific absolute paths such as:

```text
D:\...
C:\Users\...
```

and makes the project easier to reproduce on another machine.

---

# 17. Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment workflow:

```text
GitHub Repository
       |
Streamlit Community Cloud
       |
Install requirements.txt
       |
Run app.py
       |
Load saved model artifacts
       |
Public streamlit.app URL
```

Because the deployed app loads the saved artifacts, the full training notebook does not need to execute every time the application starts.

When the GitHub repository is updated, the Streamlit application can automatically redeploy from the latest repository version.

---

# 18. Reproducibility and Leakage Control

Several rules are used throughout the project to reduce data leakage and optimistic evaluation.

## Preprocessing

Scalers, imputers and encoders are fitted only on training folds.

Validation observations are transformed using rules learned from training data.

## Threshold selection

Thresholds are selected using Inner OOF predictions.

Outer Validation labels are never used to select thresholds.

## Model selection

The Outer Validation data is used only after the complete Inner-CV rule has been selected.

## Feature selection

The feature strategies are predefined before evaluating the feature-selection Nested-CV procedure.

Outer-fold results are not used to redesign the feature sets.

## Historical benchmark

The 160-row historical benchmark is used only after the final rule has been frozen.

The benchmark result does not trigger:

- feature changes
- model changes
- parameter changes
- threshold changes
- random-seed changes

## Competition data

Competition labels are not available to the training pipeline.

Leaderboard results are treated as external evaluation information rather than as part of the model-selection loop.

---

# 19. Limitations

This project is an academic machine-learning demonstration and should not be interpreted as a production banking decision system.

Important limitations include:

- The dataset is relatively small.
- The dataset is historical and may not represent modern lending populations.
- The `5:1` FN-to-FP cost ratio is a benchmark cost assumption rather than a measured modern bank loss function.
- Some variables may be sensitive or act as proxies for sensitive characteristics.
- The model has not been certified for fair-lending compliance.
- Predicted probabilities should not automatically be interpreted as calibrated real-world default probabilities.
- The historical benchmark is diagnostic rather than a fully new blind test.
- The Streamlit app is intended for educational demonstration and risk screening only.

> **The application should not be used to make real lending decisions without substantial additional validation, governance and fairness review.**

---

# 20. Future Improvements

Possible extensions include:

- Probability calibration
- SHAP-based model interpretation
- Applicant-level explanations
- More extensive hyperparameter search
- Additional cost-sensitive algorithms
- Cost-aware ensemble models
- Model calibration monitoring
- Fairness analysis
- Drift detection
- More user-friendly categorical labels in the Streamlit interface
- Authentication and role-based access
- Database integration
- Prediction logging
- REST API deployment
- Docker containerization
- Automated CI/CD testing

---

## Academic Demonstration

This repository was developed as a statistics / machine-learning project focused on understanding:

- classification
- confusion matrices
- asymmetric business cost
- probability thresholds
- preprocessing pipelines
- model comparison
- Nested Cross-Validation
- feature selection
- data leakage
- deployment

The primary goal is not simply to obtain the highest leaderboard score, but to construct a transparent and defensible modelling workflow from raw data to deployed application.
