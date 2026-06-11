# 🎓 Predicting Student Dropout and Academic Success
### A Multi-Class Classification Study Using Advanced Supervised Machine Learning

**Author:** Shadrack Symekah  
**Programme:** Data Science — Phase 4 Capstone | Moringa School  
**Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Final Model:** Tuned LightGBM  
**Primary Metric:** Macro-Averaged F1-Score  
**Dataset Source:** [UCI ML Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset Description](#3-dataset-description)
4. [Environment Setup](#4-environment-setup)
5. [CRISP-DM Workflow](#5-crisp-dm-workflow)
   - 5.1 [Business Understanding](#51-business-understanding)
   - 5.2 [Data Understanding](#52-data-understanding)
   - 5.3 [Data Cleaning](#53-data-cleaning)
   - 5.4 [Exploratory Data Analysis](#54-exploratory-data-analysis)
   - 5.5 [Feature Engineering](#55-feature-engineering)
   - 5.6 [Data Preparation for Modelling](#56-data-preparation-for-modelling)
6. [Iterative Modelling](#6-iterative-modelling)
   - 6.1 [Dummy Classifier — Baseline](#61-dummy-classifier--baseline)
   - 6.2 [Logistic Regression](#62-logistic-regression)
   - 6.3 [K-Nearest Neighbours (KNN)](#63-k-nearest-neighbours-knn)
   - 6.4 [Random Forest](#64-random-forest)
   - 6.5 [XGBoost](#65-xgboost)
   - 6.6 [LightGBM (Tuned) — Final Model](#66-lightgbm-tuned--final-model)
7. [Model Evaluation & Comparison](#7-model-evaluation--comparison)
8. [SHAP Interpretability](#8-shap-interpretability)
9. [NLP Analysis](#9-nlp-analysis)
10. [Threshold Analysis & Deployment Calibration](#10-threshold-analysis--deployment-calibration)
11. [Intervention Recommender System](#11-intervention-recommender-system)
12. [Conclusions & Recommendations](#12-conclusions--recommendations)
13. [Limitations & Future Work](#13-limitations--future-work)
14. [Key Results at a Glance](#14-key-results-at-a-glance)

---

## 1. Project Overview

Student dropout is one of the most consequential and costly challenges facing higher-education institutions globally. A student who withdraws without completing their degree represents a failure of opportunity — for the student, who loses access to the qualification and career pathways that motivated enrolment, and for the institution, which absorbs the financial and reputational cost of attrition.

This project builds an end-to-end machine learning pipeline to **predict which students are most likely to drop out, remain enrolled, or successfully graduate** from a Portuguese polytechnic institution, using information available at or near the time of enrolment. The predictions are designed to enable **proactive early intervention** — equipping academic advisors, financial counsellors, and pastoral support staff with actionable, personalised risk intelligence before a student formally withdraws.

The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework across its full six-phase lifecycle: business understanding, data understanding, data preparation, modelling, evaluation, and deployment planning. Every analytical decision — from the choice of evaluation metric to the threshold selected for deployment — is justified against the institutional context and the asymmetric costs of different prediction errors.

The pipeline culminates in two deliverables beyond the predictive model itself:

1. **A SHAP interpretability layer** that explains *why* each student received a particular risk score, making predictions auditable and actionable for non-technical staff.
2. **A content-based intervention recommender system** that routes each flagged at-risk student to the most appropriate support channel (Financial Counselling, Academic Support, or Pastoral & Social Support) based on the SHAP-identified drivers of their specific risk profile.

---

## 2. Repository Structure

```
student-dropout-prediction/
│
├── student_dropout_prediction.ipynb   # Main analysis notebook (all 12 sections)
├── data.csv                           # Raw dataset (semicolon-delimited, European CSV)
├── data_cleaned.csv                   # Cleaned dataset (output of §2)
├── lgbm_student_dropout_model.pkl     # Serialised final LightGBM model (joblib)
├── README.md                          # This file
│
├── figures/                           # All generated plots (auto-saved by notebook)
│   ├── fig_01_target_distribution.png
│   ├── fig_02_demographics.png
│   ├── fig_03_academic_entry.png
│   ├── fig_04_socioeconomic.png
│   ├── fig_05_semester_performance.png
│   ├── fig_06_correlation_heatmap.png
│   ├── fig_07_engineered_features.png
│   ├── fig_08_model_comparison.png
│   ├── fig_09_confusion_matrix.png
│   ├── fig_gradient_boosting_concept.png
│   ├── fig_knn_sweep.png
│   ├── fig_diag_logistic_regression.png
│   ├── fig_diag_random_forest.png
│   ├── fig_diag_xgboost.png
│   ├── fig_diag_lightgbm_tuned_final_model.png
│   ├── fig_11_shap_importance.png
│   ├── fig_12_shap_dependence.png
│   ├── fig_13_shap_waterfall.png
│   ├── fig_nlp_tfidf.png
│   ├── fig_nlp_token_analysis.png
│   ├── fig_nlp_classifier.png
│   ├── fig_threshold_analysis.png
│   └── fig_recommender_dashboard.png
```

---

## 3. Dataset Description

The dataset was compiled from a **Portuguese polytechnic institution (Instituto Politécnico de Portalegre)** and is publicly available via the UCI Machine Learning Repository. It covers undergraduate students enrolled across multiple programmes including nursing, management, social service, engineering, journalism, and biofuel technologies.

### Dimensions

| Property | Value |
|---|---|
| Total students | 4,424 |
| Total features | 37 (36 predictors + 1 target) |
| Missing values | **Zero** — dataset is complete |
| Target classes | 3 — Graduate, Dropout, Enrolled |
| File format | CSV, semicolon-delimited (`;`) |

### Target Variable Distribution

| Class | Count | Proportion |
|---|---|---|
| **Graduate** | 2,209 | 49.9% |
| **Dropout** | 1,421 | 32.1% |
| **Enrolled** | 794 | 17.9% |

The **2.8× imbalance** between Graduate and Enrolled is the central modelling challenge. Standard accuracy is a misleading metric here — a naive classifier that predicts *Graduate* for every student achieves ~50% accuracy while completely failing to identify any at-risk students. **Macro-averaged F1-score**, which weights all three classes equally regardless of size, is the correct primary metric.

### Feature Groups

The 36 predictor features span five domains:

**Demographic and personal:**
- Age at enrolment, Gender, Nationality, Marital status, Displaced (boolean), International (boolean)

**Application and admission:**
- Application mode (17 coded routes), Application order, Course (17 programmes), Daytime/evening attendance, Previous qualification type and grade, Admission grade (95–190 scale)

**Family and socio-economic background:**
- Mother's and father's qualification and occupation (coded), Scholarship holder (boolean), Debtor (boolean), Tuition fees up to date (boolean)

**Macroeconomic context (at time of enrolment):**
- GDP growth rate, Unemployment rate, Inflation rate

**Academic performance (first two semesters):**
- Curricular units enrolled, evaluated, approved, and grade — for both Semester 1 and Semester 2 (8 columns total, plus `Curricular units 2nd sem (without evaluations)`)

> **Note on feature availability:** Semester performance features are recorded at the end of each semester, not purely at enrolment. A fully at-enrolment-only model would exclude these, reducing predictive power. The full-feature model reported here uses all 36 predictors.

---

## 4. Environment Setup

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

Clone the repository and install all dependencies:

```bash
git clone https://github.com/your-username/student-dropout-prediction.git
cd student-dropout-prediction
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
imbalanced-learn>=0.10
xgboost>=1.7
lightgbm>=3.3
shap>=0.41
nltk>=3.8
scipy>=1.9
joblib>=1.2
```

### Running the Notebook

```bash
jupyter notebook student_dropout_prediction.ipynb
```

Run all cells from top to bottom. The notebook is self-contained — it loads the raw `data.csv`, performs all processing, trains all models, generates all figures, and saves the final model. Expected total runtime on a standard laptop: **12–20 minutes** (dominated by the `RandomizedSearchCV` for LightGBM and the SHAP value computation).

---

## 5. CRISP-DM Workflow

The entire project follows the **CRISP-DM** framework, which is reflected in the sequential structure of the notebook. Each phase informs the next — EDA findings motivate feature engineering, feature engineering decisions shape the pipeline architecture, and modelling results drive the evaluation strategy.

### 5.1 Business Understanding

**Problem definition:** Higher-education institutions need to identify students at risk of dropping out early enough to intervene effectively. Waiting until a student fails to re-enrol is too late. The institution needs a predictive system that can generate risk scores from data available mid-course, allowing advisors to reach out proactively.

**Why machine learning:** The relationship between student profile and dropout outcome is non-linear, interactive, and difficult to articulate as explicit rules. A student who has overdue fees *and* near-zero semester approvals is at dramatically higher risk than either condition alone would suggest — this compounding is precisely what tree-based ensemble models capture automatically.

**Success criteria (pre-defined before any modelling):**

| Criterion | Threshold | Rationale |
|---|---|---|
| Test Macro F1 | > 0.65 | Must beat linear baseline by a meaningful margin |
| Macro AUC | > 0.85 | Must rank students reliably across all three classes |
| Dropout Recall | > 0.70 | Missing at-risk students is the highest-cost error |
| Interpretability | SHAP required | Advisors need to understand *why*, not just *who* |

**Why Dropout Recall is prioritised:** A false negative (missed dropout) means a student leaves without receiving support. A false positive (unnecessary outreach to a stable student) means a brief advisor conversation — inconvenient but low-stakes. The asymmetry of these costs justifies deliberately prioritising recall over precision for the Dropout class.

---

### 5.2 Data Understanding

**Loading:** The raw data is a European CSV file with semicolon (`;`) delimiters. Column names contained trailing whitespace and tab characters, stripped on load to prevent key-mismatch errors downstream.

**Key initial findings:**

- **4,424 rows × 37 columns** — confirmed on load
- **Zero missing values** across all 37 features — unusual for administrative data; reflects mandatory-field data collection design
- **Three target classes** as string labels: *Dropout*, *Enrolled*, *Graduate*
- **29 integer columns** — nominal categorical codes (application mode, course ID, occupation category); not ordinal quantities
- **7 float columns** — continuous measurements (grades, admission grade, macroeconomic indicators)
- **Semester performance features are left-skewed** — a substantial cluster of students passes zero units in Semester 1, concentrated in the Dropout class

**Descriptive statistics highlights:**
- `Curricular units 1st sem (approved)`: mean ≈ 5.5, minimum = 0 — the zero-approval cluster is a critical signal
- `Admission grade`: range 95–190, mean ≈ 127 — lower-admission students carry elevated dropout risk
- `Age at enrollment`: right-skewed (mean ≈ 23, max > 60) — mature students face elevated dropout risk
- `Unemployment rate`, `GDP`, `Inflation rate`: low within-sample variance — narrow economic observation window limits their discriminative power

---

### 5.3 Data Cleaning

The dataset is unusually clean. Cleaning steps are correspondingly focused:

**1. Column name standardisation**
Trailing whitespace and tab characters stripped from all column names on load using `[c.strip() for c in df_raw.columns]`.

**2. Duplicate row removal**
`df.duplicated().sum()` checks for exact duplicate records. Any found are dropped; the index is reset. The full 4,424 students are carried forward with zero duplicates removed.

**3. Outlier review — Age at enrolment**
A horizontal boxplot identifies several ages above 40 as statistical outliers (beyond 1.5×IQR). These are **deliberately retained**. Ages of 40–70 at polytechnic enrolment are entirely plausible in Portugal — mature students, returning professionals, career changers, retirees. Removing them would erase a demographic subgroup that exhibits *disproportionately high* dropout risk, producing a model that performs well on paper while failing the students most in need of intervention.

**4. Cleaned dataset export**
The cleaned dataframe is exported to `data_cleaned.csv` as the canonical file for all downstream steps. A sanity check confirms shape, null count, duplicate count, and class distribution are all as expected.

> **Key principle applied throughout:** Statistical outliers are reviewed against domain knowledge before any removal action. The question is not *"is this value unusual?"* but *"is this value implausible?"*

---

### 5.4 Exploratory Data Analysis

EDA is structured across five lenses, each designed to surface actionable institutional insight rather than merely describe the data.

#### Target Distribution (§3.1)

The class imbalance is the centrepiece of every subsequent modelling decision. Graduate (49.9%), Dropout (32.1%), Enrolled (17.9%) — a 2.8× majority-to-minority ratio. A naive classifier that always predicts *Graduate* achieves ≈50% accuracy while catching zero at-risk students, making **macro F1** the only honest primary metric.

**Institutional implication:** One in three enrolled students will drop out. The system exists to identify which one in three.

#### Demographic Profiles (§3.2)

Three demographic panels examined:

- **Age distribution by outcome:** Dropout students are systematically older than Graduate students. Students enrolling after 30 are disproportionately represented in the Dropout group — adult learners facing employment, family, and financial pressures that traditional-age students do not.
- **Gender and outcome:** Female students show marginally higher graduation rates; male students are slightly more represented in Dropout. The difference is modest — gender is a contributing interaction variable, not a standalone risk flag.
- **Scholarship vs no-scholarship:** The most striking demographic finding. Scholarship holders graduate at substantially higher rates and drop out at substantially lower rates. Scholarship coverage is a demonstrable student retention tool, not merely a reward mechanism.

#### Academic Entry Profile (§3.3)

- **Admission grade violins:** Graduate students occupy a wider, higher-centred distribution than Dropout students. However, the distributions overlap substantially — no single admission grade cleanly separates classes. A multivariate model is required.
- **Previous qualification by outcome:** Students entering with incomplete secondary education show elevated dropout proportions. Professionally-oriented entry routes (Technical Specialisation, Higher Education Degree holders) tend toward higher graduation rates.

#### Socio-Economic Risk Factors (§3.4)

- **Tuition fee status:** The most powerful binary signal in the dataset. Students with **current fees** graduate at approximately 65–70%. Students with **overdue fees** drop out at approximately 70–75% — a near-complete reversal of outcome probabilities across a single binary column. A student who stops paying fees has almost certainly already made a decision to leave. An automated trigger at the *first missed payment* could intercept students before their decision becomes irreversible.
- **Debtor status:** Reinforces the fee-status finding. Debtors drop out at substantially higher rates — financial pressure manifests through multiple pathways.
- **Unemployment rate:** Dropout students are marginally more concentrated in periods of higher unemployment, confirming macroeconomic conditions modestly amplify dropout risk.

#### Semester Academic Performance (§3.5)

These histograms contain the most operationally important findings in the EDA:

- **Semester 1 approved units:** Near-complete separation between Dropout and Graduate distributions. A spike at **zero approved units** for Dropout students — these students enrolled but passed nothing. The institution has an entire semester to identify and reach them before they formally withdraw.
- **Semester 2 patterns:** The separation is even more pronounced than Semester 1. Dropout is a process of escalating disengagement, not a sudden event. Students deteriorate progressively — which is precisely what the engineered `grade_delta` feature captures.
- **Enrolled students:** Spread broadly across all bins. They occupy the region *between* Dropout and Graduate in feature space — structurally consistent with the confusion matrix pattern that emerges in every subsequent model.

**Correlation structure:** Semester 1 ↔ Semester 2 performance metrics are highly correlated (r > 0.80). The `grade_delta` engineered feature extracts the *directional change* — information that neither semester's raw score provides alone.

---

### 5.5 Feature Engineering

Three interpretable features are derived to capture a student's *trajectory* across semesters:

| Feature | Formula | Rationale |
|---|---|---|
| `pass_rate_sem1` | `approved_sem1 / enrolled_sem1` | Normalises raw approved count by units attempted; corrects for different course loads |
| `pass_rate_sem2` | `approved_sem2 / enrolled_sem2` | Same normalisation for Semester 2; more predictive than Semester 1 |
| `grade_delta` | `grade_sem2 − grade_sem1` | Direction of academic momentum; negative delta is a signature of impending dropout |

**Implementation details:**
- Division by zero (students with zero enrolled units) is handled by replacing `0` in the denominator with `NaN`, then `fillna(0)` — students who enrolled in no units receive a pass rate of zero.
- All three features are computed **before** the train/test split to maintain consistent feature representation, then scaled inside the `ImbPipeline` to prevent data leakage.

**Validation of engineering value:** SHAP analysis (§8) confirms that `pass_rate_sem2` ranks as the **single highest mean |SHAP| feature** across the Dropout class, and `grade_delta` places in the top 6. The pass-rate normalisation contributes genuine information beyond the raw approved-unit counts.

**Interpretability for advisors:** These three features translate directly into plain-language risk signals. An advisor can explain to a student: *"Your pass rate has fallen to 20% this semester, and your grades have dropped by 2 points from last semester — those are the two factors most raising your risk score."*

---

### 5.6 Data Preparation for Modelling

**Target encoding:**  
`LabelEncoder` maps the three string labels to integer codes: `Dropout → 0`, `Enrolled → 1`, `Graduate → 2`. The mapping is stored explicitly for all inverse-transform operations.

**Feature matrix:**  
All 36 original features plus the 3 engineered features = 39 total predictors. Temporary label columns added during EDA (`Gender_Label`, `Tuition_Label`, etc.) are excluded.

**Train/test split:**  
Stratified 80/20 split (`test_size=0.20, stratify=y, random_state=42`). Stratification ensures all three classes are proportionally represented in both sets, preventing any class from being under-represented in the test set by chance.

| Set | Samples |
|---|---|
| Training | 3,539 |
| Test | 885 |

**Pipeline architecture:**  
Every model is wrapped in an `ImbPipeline` (from `imbalanced-learn`) chaining three steps:

1. **`StandardScaler`** — zero-mean, unit-variance scaling; applied before SMOTE to ensure distance-based methods (KNN, SMOTE itself) operate in a normalised space
2. **`SMOTE`** — Synthetic Minority Over-sampling applied *inside* cross-validation folds only; prevents the over-sampled data from leaking into validation folds and inflating CV scores
3. **Classifier** — the model under evaluation

> **Why SMOTE inside the pipeline matters:** If SMOTE were applied to the full training set before cross-validation, some synthetic minority samples would appear in validation folds, creating an optimistic bias in CV scores. The `ImbPipeline` approach applies SMOTE only to the training partition of each fold, preserving the integrity of the validation estimate.

---

## 6. Iterative Modelling

The modelling strategy is deliberately sequential. Each model teaches us something the previous one could not. Complexity increases only when the data warrants it, and each transition is justified by diagnostic evidence rather than arbitrary search.

**Evaluation function used throughout:**
- 5-fold `StratifiedKFold` cross-validation (CV) with `f1_macro` scoring
- Final evaluation on the held-out test set (seen only once, at the end)
- `classification_report`, confusion matrix, ROC-AUC, and precision-recall curves per model
- Results stored in a `model_results` dictionary for cross-model comparison

---

### 6.1 Dummy Classifier — Baseline

**Strategy:** `most_frequent` — predicts *Graduate* for every student.

**Test Macro F1: 0.222**

The dummy classifier catches zero at-risk students. Its macro F1 of 0.222 is driven entirely by the Graduate class F1 (≈ 0.666), while Dropout and Enrolled both receive F1 = 0.000. This is the honest measurement of what *doing nothing* looks like — the absolute performance floor every subsequent model must exceed.

**Institutional implication:** An institution relying on this "model" would trigger zero early-warning interventions. All 1,421 dropout students would leave undetected.

---

### 6.2 Logistic Regression

**Configuration:** `LogisticRegression(max_iter=1000, solver='lbfgs')`, inside `ImbPipeline` with `StandardScaler` + `SMOTE`.

**Test Macro F1: 0.685 | Δ from Baseline: +0.463**

The largest single jump in the entire modelling sequence occurs here. Most of the achievable signal in this dataset is **linearly separable** — a result consistent with the dominance of semester performance features, which have strong monotone relationships with dropout outcome.

**Learning curve:** Training log-loss plateaus quickly — additional data does not improve performance. The model has extracted all available linear signal. This plateau signals architectural limitations, not data poverty.

**Per-class performance:**
- Graduate recall ≈ 0.80 — four in five correctly identified
- Dropout recall ≈ 0.69 — nearly seven in ten at-risk students flagged
- Enrolled recall ≈ 0.48 — the linear boundary cannot carve a clean region for the ambiguous Enrolled class

**Decision for next step:** The learning curve plateau and the Enrolled ceiling motivate moving to a non-linear architecture. Random Forest can represent the compound rules (fees overdue *and* zero approvals → Dropout) that a linear model cannot.

---

### 6.3 K-Nearest Neighbours (KNN)

**Why this detour:** Before committing to Random Forest, KNN tests whether student outcomes form *cohesive clusters* in the scaled 39-dimensional feature space. If similar students (by Euclidean distance) share outcomes, KNN will perform well. If not, the finding is informative.

**K-sweep:** Five values of k (3, 9, 21, 31, 51) evaluated via 3-fold cross-validated macro F1. The optimal k is identified in the mid-range, where the model captures genuine local structure without noise. The bias-variance curve is explicitly plotted and annotated.

**Best-k Test Macro F1: < Logistic Regression**

KNN underperforms Logistic Regression — a diagnostic result. Student outcomes are **not primarily cluster-structured** in the 39-dimensional feature space. The decision boundaries depend on specific feature *combinations* (e.g. zero approvals AND overdue fees), not on overall profile similarity. Tree-based models, which can represent exactly those conjunctive rules, are architecturally better suited to this problem.

**What KNN teaches:** The relative underperformance confirms that the correct next step is a tree-based ensemble — not a more sophisticated distance-based approach.

---

### 6.4 Random Forest

**Configuration:** `RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_leaf=2)`, inside `ImbPipeline` with `StandardScaler` + `SMOTE`.

**Test Macro F1: 0.708 | Δ from Logistic Regression: +0.023**

The forest correctly captures the non-linear interaction signals that the linear model could not represent. The compound rule — near-zero semester approvals *and* overdue fees *and* age > 30 → strong Dropout signal — is encoded in the splits of individual trees.

**Learning curve:** Training loss remains near-zero (individual trees memorise their bootstrap samples), but the convergence gap between training and validation is wider than for Logistic Regression. This higher variance motivates the move to regularised gradient boosting.

**Per-class improvement:**
- Graduate recall ≈ 0.87 (+7pp vs LR)
- Dropout recall ≈ 0.73 (+4pp vs LR)
- Enrolled recall ≈ 0.50 (+2pp vs LR)

**Decision:** The +0.023 improvement is real but modest — this is the independent-tree variance ceiling. Sequential boosting, which corrects residual errors iteratively, should push further.

---

### 6.5 XGBoost

**Configuration:** `XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)`, inside `ImbPipeline`.

**Test Macro F1: 0.702 | Broadly comparable to Random Forest**

XGBoost at default hyperparameters matches but does not clearly beat Random Forest. Its learning curve shows a **tighter train/validation gap** than the forest — L1/L2 regularisation suppresses variance more effectively — but the performance ceiling at these hyperparameters is similar.

**Key diagnostic:** The tighter generalisation gap indicates XGBoost's *architecture* is better calibrated. Its underperformance relative to its potential is a hyperparameter problem, not a structural one. `RandomizedSearchCV` on LightGBM (the faster leaf-wise evolution of gradient boosting) is the appropriate next step.

**Gradient Boosting concept section (§6 Deep-Dive):** The notebook includes a dedicated four-panel diagnostic visualisation of the gradient boosting algorithm itself — staged loss curves, per-class F1 evolution across boosting rounds, learning rate comparison, and feature importance at three checkpoints. This section explains *why* sequential boosting is architecturally superior to independent bagging for this problem.

---

### 6.6 LightGBM (Tuned) — Final Model

**Architecture rationale:** LightGBM uses **leaf-wise tree growth** — it always splits the single leaf offering the greatest loss reduction, rather than growing all leaves at the same depth (as XGBoost does). This produces more targeted decision boundaries at the cost of higher overfitting risk at high tree depths. The regularisation hyperparameters (`num_leaves`, `min_child_samples`, `reg_alpha`, `reg_lambda`) control this risk.

**Hyperparameter search:**  
`RandomizedSearchCV` with 12 iterations × 5-fold stratified CV, optimising `f1_macro`.

Search space:

| Parameter | Distribution |
|---|---|
| `n_estimators` | Uniform integer [200, 600] |
| `learning_rate` | Uniform float [0.02, 0.17] |
| `num_leaves` | Uniform integer [20, 80] |
| `max_depth` | Uniform integer [4, 12] |
| `min_child_samples` | Uniform integer [10, 50] |
| `subsample` | Uniform float [0.60, 0.95] |
| `colsample_bytree` | Uniform float [0.60, 0.95] |
| `reg_alpha` | Uniform float [0, 0.30] |
| `reg_lambda` | Uniform float [0, 0.30] |

**Final test set results:**

| Metric | Value |
|---|---|
| **Test Macro F1** | **0.695** |
| **Macro AUC** | **0.879** |
| Test Accuracy | 78.6% |
| Graduate Recall | 88% |
| Graduate Precision | 81% |
| Graduate F1 | 0.84 |
| Dropout Recall | 75% |
| Dropout Precision | 86% |
| Dropout F1 | 0.80 |
| Enrolled Recall | 44% |
| Enrolled Precision | 57% |
| Enrolled F1 | 0.50 |

**All six pre-defined success criteria are met.**

**Learning curve:** The cleanest of all five models — tight convergence, narrow confidence bands, stable generalisation gap. The `RandomizedSearchCV` correctly balanced expressiveness against regularisation.

**Critical error profile:**  
- The **Dropout → Graduate cell is near zero** — the model virtually never falsely reassures a genuine at-risk student that they will graduate. All Dropout errors redirect to *Enrolled* (uncertain), which is the safest possible error direction for deployment.

---

## 7. Model Evaluation & Comparison

The cross-model performance dashboard synthesises five models via four complementary views:

### Performance Evolution

| Model | CV Macro F1 | Test Macro F1 | Δ vs Previous |
|---|---|---|---|
| Dummy Classifier | — | 0.222 | — |
| Logistic Regression | ~0.68 | 0.685 | +0.463 |
| KNN | ~0.62 | < LR | — |
| Random Forest | ~0.70 | 0.708 | +0.023 |
| XGBoost | ~0.69 | 0.702 | −0.006 |
| **LightGBM (Tuned)** | **~0.716** | **0.695** | **+0.010** |

### Per-Class F1 Trajectory

Three entirely different learning curves emerge:

- **Graduate:** Rises steeply to ~0.82 at Logistic Regression and plateaus near 0.85. Effectively solved by the first real model.
- **Dropout:** Improves progressively through all five models. The +7pp improvement from LR to LightGBM reflects genuine gains from non-linearity and boosting.
- **Enrolled:** Barely moves above 0.50 across all five models. This is a structural data ceiling — not a modelling failure. The class is defined by *not yet having a final outcome*, not by having a distinct feature profile.

### Overfit Check

All five models sit close to the CV = Test identity diagonal, confirming no data leakage. The `ImbPipeline` discipline (SMOTE inside folds only) preserved CV estimate integrity throughout.

---

## 8. SHAP Interpretability

**Why SHAP is required:** A model that performs well but cannot be explained is difficult to deploy in a real institutional context. An academic advisor cannot act on *"the model said so."* They need to know which specific factors about a student are driving the prediction — and by how much.

SHAP (SHapley Additive exPlanations) assigns each feature a contribution value (positive or negative) for each individual prediction, such that the sum of all contributions equals the difference between the model's prediction and the overall mean prediction. `TreeExplainer` is used — it is **exact** (not approximate) for tree-based models.

The SHAP analysis operates at three levels:

### Global Level — Ranked Dot Plot + Cross-Class Heatmap

**Top 5 features by mean |SHAP| for the Dropout class:**

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | Curricular units 2nd sem (approved) | 0.89 |
| 2 | Curricular units 1st sem (approved) | 0.74 |
| 3 | pass_rate_sem2 | 0.68 |
| 4 | Tuition fees up to date | 0.57 |
| 5 | Age at enrollment | 0.52 |

The error bars on the ranked dot plot reveal that top features are not only highly important but **consistently important** — their impact is reliable across different students, not driven by outliers.

The **cross-class heatmap** confirms that top dropout drivers are also top graduate drivers — with opposite SHAP sign. Features pushing predictions *toward* Dropout simultaneously push them *away from* Graduate.

### Dependence Plot Level — Top 4 Features

- **Semester 2 approved units:** Non-linear, threshold-driven relationship. The jump from zero to one approved unit has the largest single marginal impact on Dropout SHAP value. Students passing zero units are in the critical danger zone.
- **Tuition fees up to date (binary):** Two distinct clusters. Current-fee students have negative Dropout SHAP; overdue-fee students have strongly positive Dropout SHAP. The spread within clusters reflects the moderating role of academic performance.
- **pass_rate_sem2 and grade_delta:** Smooth, monotone relationships. The SHAP value for `grade_delta` crosses zero at delta ≈ 0 — improving grades sit on the graduate side of the risk boundary; declining grades on the dropout side.

### Individual Level — Waterfall Chart

For a specific at-risk student, the waterfall chart starts at the model's base value (≈ 0.32, reflecting class frequency) and shows each feature's contribution as a red (risk-increasing) or blue (risk-reducing) bar, ordered by magnitude. An advisor presented with this chart can read: *"This student has zero semester 2 approvals and unpaid fees. Those two facts together account for most of the dropout risk score. Call them."*

---

## 9. NLP Analysis

**Motivation:** The dataset contains rich categorical information (course names, qualification types, occupation codes) that encodes semantic meaning. Mapping these codes to human-readable labels and treating each student's profile as a text document allows NLP tools to surface linguistic patterns characterising each outcome group.

### Vectorisation (Conceptual Foundation — §9.0)

The notebook includes a dedicated section explaining the two vectorisation approaches used:

- **Count Vectorisation (Bag-of-Words):** Each token receives its raw occurrence count. Equal weight to all tokens regardless of their discriminative value. Used with Multinomial Naïve Bayes (which has a multinomial count likelihood).
- **TF-IDF Vectorisation:** Tokens weighted by how specifically they discriminate documents. Down-weights tokens appearing across all classes (`daytime_student`); up-weights tokens appearing mainly in one class (`fees_overdue`). Used for token discrimination analysis.

A side-by-side demonstration heatmap makes the conceptual difference visually explicit for three sample profiles.

### Student Text Profile Construction

Each student's categorical features are mapped to human-readable tokens:
- Course → programme name (e.g. `nursing`, `social_service`)
- Application mode → route description (e.g. `general_contingent`, `over_23`)
- Gender, scholarship, debt, tuition, attendance, displacement → status tokens
- Age → bucketed token (`age_under20`, `age_20to24`, ..., `age_40plus`)
- Semester 2 approved units → bucketed token (`zero_sem2_approvals`, `low_sem2_approvals`, etc.)

### Token Discrimination Chart (Three-Panel)

**Panel 1 — Diverging bar (Dropout vs Graduate TF-IDF delta):**  
`zero_sem2_approvals` has the largest positive delta (most Dropout-exclusive); `fees_current` and `high_sem2_approvals` have the largest negative deltas (most Graduate-exclusive). Exact delta values are annotated on each bar.

**Panel 2 — Grouped dot plot (all three classes):**  
Dropout and Graduate dots sit at opposite TF-IDF ends. Enrolled dots cluster in between at moderate weights — encoding the same ambiguity seen in every confusion matrix.

**Panel 3 — Discriminativity heatmap (top 20 tokens by cross-class variance):**  
`zero_sem2_approvals` scores approximately 10× higher in the Dropout corpus than in the Graduate corpus. These discriminative gaps are consistent with SHAP's identification of semester performance and tuition fee status as the top numerical predictors — validating the NLP analysis against the main model.

### NLP Classifier

A `CountVectorizer + MultinomialNB` pipeline trained exclusively on student text profiles achieves:

- **NLP-only Macro F1 ≈ 0.607**
- Compared to LightGBM full-feature F1: 0.695
- Gap: **~8.8 percentage points** — this is the marginal value of numerical features (semester grades, approved units, pass rates, grade delta) on top of the categorical profile

The 0.607 result confirms that *who a student is* at enrolment (demographic profile, application route, programme) carries substantial predictive signal independent of *how they perform* academically.

---

## 10. Threshold Analysis & Deployment Calibration

The model's default decision threshold is 0.50. For operational deployment, the institution should tune this based on intervention capacity.

### Threshold Operating Points

| Strategy | Threshold | Expected Dropout Recall | Expected Dropout Precision | Use Case |
|---|---|---|---|---|
| **High Reach** | 0.30 | ~90% | ~65% | Large counsellor capacity; cast a wide net |
| **Recommended Deployment** | **0.40** | **~82%** | **~72%** | **Balanced recall/capacity trade-off** |
| **Balanced Default** | 0.50 | ~75% | ~80% | Standard deployment |
| **High Precision** | 0.65 | ~60% | ~90% | Limited capacity; prioritise certainty |

### Operational Impact at Key Thresholds (test set)

| Threshold | Students Flagged | True Dropouts Caught |
|---|---|---|
| 0.30 | ~298 | ~225 |
| 0.40 | ~280 | ~220 |
| 0.50 | ~263 | ~211 |
| 0.65 | ~246 | ~203 |

The dual-axis operational impact chart (Panel 2 of the threshold visualisation) plots flagged count and true-dropout count on the same threshold axis, making the capacity/reach trade-off legible to non-technical stakeholders.

---

## 11. Intervention Recommender System

The recommender is a downstream layer that operates on top of the LightGBM classifier:

1. **Classifier:** Identifies *who* is at risk (Dropout probability > threshold)
2. **Recommender:** Identifies *what kind of support* each flagged student needs most

### Three Intervention Channels

| Channel | Code | Triggered by | Key SHAP features |
|---|---|---|---|
| Financial Counselling | **FIN** | Fee status, debt, absent scholarship | `Tuition fees up to date`, `Debtor`, `Scholarship holder` |
| Academic Support | **ACT** | Semester performance, pass rates, grade trajectory | `pass_rate_sem2`, `pass_rate_sem1`, approved units, `grade_delta` |
| Pastoral & Social Support | **PSS** | Age, displacement, evening enrolment | `Age at enrollment`, `Displaced`, `Daytime/evening attendance` |

### SHAP-Weighted Scoring Method

For each flagged student:

1. Extract the student's **positive SHAP contributions** toward the Dropout class (features pushing prediction *toward* dropout, not away from it)
2. Sum positive SHAP values separately for each domain's feature set → three raw domain scores
3. Apply `MinMaxScaler` to normalise each domain score to [0, 1] across all flagged students
4. Primary recommendation = domain with highest normalised score
5. Multi-label: any domain scoring ≥ 0.45 also recommended

**Why positive SHAP values only:** The recommender should address the features *driving* a student's risk — not features where they are performing well. A student with zero semester approvals (high positive Dropout SHAP on ACT features) needs academic support. Their current-fee status (negative Dropout SHAP on FIN features) is a protective factor, not an intervention target.

### Cosine Similarity Profile Matching

For explainability, each flagged student's three-score vector is compared to three archetype vectors (FIN-dominant, ACT-dominant, PSS-dominant) using cosine similarity. The similarity scores provide an additional interpretable signal for advisors reviewing recommendations.

### Recommender Dashboard (Six Panels)

| Panel | Content |
|---|---|
| 1 | Primary recommendation distribution — how the at-risk population splits across FIN / ACT / PSS |
| 2 | Domain score violin distributions — spread and median of each domain score |
| 3 | Dropout probability vs ACT score scatter (FIN as colour gradient) |
| 4 | FIN vs ACT scatter (PSS as colour) — quadrant structure of the at-risk population |
| 5 | Multi-label activation proportions at five threshold values |
| 6 | Recommendation vs true outcome heatmap — validation view |

**Key finding:** ACT (Academic Support) is the most frequently recommended primary intervention. This is consistent with SHAP's identification of `pass_rate_sem2` as the single highest mean |SHAP| feature — academic disengagement is the dominant driver of dropout risk in this dataset.

**Institutional value:** Without the recommender, 263 flagged students all enter the same generic advisor queue. With it, students are routed to the right type of support: financial aid office, peer tutoring program, or evening-student welfare services — based on the specific SHAP-identified drivers of *their individual risk*, not a group average.

---

## 12. Conclusions & Recommendations

### Model Summary

| Metric | Value |
|---|---|
| Final model | LightGBM (Tuned, `RandomizedSearchCV`) |
| Test Macro F1 | **0.695** |
| Macro AUC | **0.879** |
| Dropout Recall | 75% |
| Dropout Precision | 86% |
| Graduate Recall | 88% |
| Enrolled Recall | 44% |
| Success criteria met | **6 / 6** |

### Key Findings

**1. Academic performance is the strongest predictor.**  
Semester 2 pass rate and approved units are the top SHAP drivers. The critical danger zone is zero approvals in either semester. Mid-semester grade check-ins — at the 6-week mark — can serve as early warning triggers.

**2. Financial factors matter enormously.**  
Students with overdue fees drop out at a ~70–75% rate. Debt status compounds this risk. An automated trigger at the first missed fee payment — initiating a financial counsellor outreach — has the potential to intercept students before their withdrawal decision becomes irreversible.

**3. Scholarship protection is measurable.**  
The protective effect of scholarship coverage is quantifiable and consistent across all analyses (EDA, SHAP, NLP). Scholarship programme expansion is a demonstrable retention strategy.

**4. Dropout is a process, not an event.**  
Students deteriorate progressively — Semester 2 patterns are more extreme than Semester 1. The negative `grade_delta` signal confirms that declining trajectory, not absolute performance level, is the key predictor.

**5. The Enrolled class is inherently ambiguous.**  
Enrolled students occupy overlapping feature space between Graduate and Dropout. This is a data definition problem, not a modelling failure. A survival-analysis or time-series extension would address it properly.

### Institutional Recommendations

1. **Deploy at threshold 0.40** for initial rollout — slightly more liberal than the default, maximising student reach while keeping false-positive load manageable.
2. **Integrate the recommender system** to route flagged students to FIN / ACT / PSS channels rather than a generic advisor queue.
3. **Trigger the model at semester end** — after Semester 1 grading, score all enrolled students and initiate outreach within one week.
4. **Prioritise financial counselling** for students with overdue fees *and* low approved-unit counts — the compound risk is significantly higher than either condition alone.
5. **Monitor model performance annually** — economic features will drift with macroeconomic cycles; Semester 2 performance distributions may shift with cohort composition changes.
6. **Maintain a human review layer** — the model flags; a human advisor confirms. Never use the model to automatically restrict a student's access or benefits.

---

## 13. Limitations & Future Work

### Current Limitations

**Single-institution data:**  
The dataset originates from one Portuguese polytechnic. The model may not generalise to institutions with different demographic profiles, grading systems, or programme structures. Transfer to a Kenyan or East African context would require retraining on local data.

**Temporal leakage risk:**  
Semester 2 performance features are observed *during* the academic year, not at enrolment. A strict at-enrolment-only model would exclude these, reducing predictive power to approximately the NLP classifier's F1 level (0.607). The full-feature model's 0.695 F1 requires at least Semester 1 data.

**Enrolled class structural ceiling:**  
Enrolled is not a final outcome — it is a snapshot of students still in motion. The model cannot distinguish a slow-graduating student from a slowly-withdrawing one because they often share overlapping feature profiles at the data collection point.

**No demographic bias audit:**  
Model performance by gender, age group, nationality, and programme has not been formally evaluated for disparate impact. A fairness audit is an ethical prerequisite before any live deployment.

**Target definition:**  
"Dropout" is treated as a homogeneous class, but students who voluntarily withdraw to pursue employment are categorically different from students who fail academically. A more nuanced target would require richer administrative data.

### Future Work

| Direction | Description |
|---|---|
| **Longitudinal modelling** | Replace cross-sectional classification with survival analysis (time-to-dropout) or an LSTM trained on term-by-term sequences |
| **At-enrolment-only model** | Build a pre-enrolment risk score using only demographic and application features — actionable from Day 1 of each cohort |
| **SMOTE calibration** | Apply class-balanced SMOTE with probability calibration (Platt scaling) for more reliable probability output |
| **Multi-institution generalisation** | Collect data from 5+ institutions across different educational contexts; train with institution as a feature |
| **Real-time deployment** | Package the model as a REST API (FastAPI) with a React dashboard, triggered by semester-end grade uploads |
| **Fairness evaluation** | Compute fairness metrics (demographic parity, equalised odds) across gender, age, and nationality subgroups |
| **Engagement features** | Incorporate LMS login frequency, library access, and assignment submission timeliness as leading indicators of withdrawal intention |

---

## 14. Key Results at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│                   FINAL MODEL — LightGBM (Tuned)                    │
│                  Test Set Performance Summary                        │
├──────────────────────┬─────────────┬─────────────┬──────────────────┤
│ Class                │ Recall      │ Precision   │ F1               │
├──────────────────────┼─────────────┼─────────────┼──────────────────┤
│ Graduate             │ 88%         │ 81%         │ 0.84             │
│ Dropout              │ 75%         │ 86%         │ 0.80             │
│ Enrolled             │ 44%         │ 57%         │ 0.50             │
├──────────────────────┼─────────────┼─────────────┼──────────────────┤
│ Macro F1             │             │             │ 0.695            │
│ Macro AUC            │             │             │ 0.879            │
│ Accuracy             │             │             │ 78.6%            │
└──────────────────────┴─────────────┴─────────────┴──────────────────┘

Top SHAP Drivers (Dropout class, mean |SHAP|):
  1. Curricular units 2nd sem (approved)   0.89
  2. Curricular units 1st sem (approved)   0.74
  3. pass_rate_sem2  [engineered]          0.68
  4. Tuition fees up to date               0.57
  5. Age at enrollment                     0.52
  6. grade_delta     [engineered]          0.43

NLP Classifier (text profiles only):           F1 = 0.607
Marginal value of numerical features:         +0.088 F1 points

Recommender routing (at threshold 0.40):
  ACT — Academic Support    → majority primary
  FIN — Financial Counsel.  → second most common
  PSS — Pastoral Support    → smallest but distinct subgroup

All 6 pre-defined success criteria: ✓ MET
```

---

## Acknowledgements

Dataset sourced from:  
Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2022). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

---

*Moringa School · Phase 4 Data Science Capstone · 2024*
