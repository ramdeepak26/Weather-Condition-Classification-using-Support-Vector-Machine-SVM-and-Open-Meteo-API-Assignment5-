# Employee Attrition Prediction — Decision Tree vs Random Forest

## Objective
Predict whether an employee is likely to leave the organization
("Attrition") using demographic, professional, and work-related
attributes, and compare a Decision Tree Classifier against a Random
Forest Classifier (100 estimators) on accuracy, precision, recall, and
F1-score.

## Dataset
- **Name:** IBM HR Analytics Employee Attrition & Performance
- **Source:** [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size:** 1,470 employees × 35 columns
- **Target variable:** `Attrition` (Yes / No)

> **Note on the data file used to run this repo:** this environment
> could not reach Kaggle to download the CSV directly, so
> `generate_demo_data.py` builds a synthetic stand-in
> (`demo_HR_Employee_Attrition.csv`) with the **exact same 35 columns
> and value ranges** as the real file, plus realistic attrition
> drivers (overtime, low satisfaction, low income, frequent travel,
> etc.) baked in. `Assignment-5.py` automatically prefers the real
> Kaggle file (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) if you drop it
> into the same folder — no code changes needed. All results below
> were produced by actually running the pipeline on the demo data.

## Libraries Used
- `pandas`, `numpy` — data loading & manipulation
- `scikit-learn` — `train_test_split`, `LabelEncoder`,
  `DecisionTreeClassifier`, `RandomForestClassifier`, evaluation metrics
- `matplotlib`, `seaborn` — confusion matrices, feature importance and
  metric comparison charts

## Methodology
1. **Data Understanding** — loaded the CSV, inspected `head()`,
   `info()`, `describe()`, and separated columns into numerical
   features, categorical features, and the target (`Attrition`).
2. **Data Preprocessing**
   - Checked for missing values (none found).
   - Dropped non-informative columns: `EmployeeCount`, `EmployeeNumber`,
     `Over18`, `StandardHours` (constants / unique IDs, no predictive
     value).
   - Label-encoded the target and all categorical columns
     (`BusinessTravel`, `Department`, `EducationField`, `Gender`,
     `JobRole`, `MaritalStatus`, `OverTime`).
   - Split data 80% train / 20% test with stratification on the
     target.
3. **Model Development**
   - Model 1: `DecisionTreeClassifier(random_state=42)`
   - Model 2: `RandomForestClassifier(n_estimators=100, random_state=42)`
   - Both trained on the same training split and evaluated on the same
     held-out test split.
4. **Evaluation** — accuracy, precision, recall, F1-score, confusion
   matrices for both models, and a feature-importance plot for the
   Random Forest.

## Results

| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|---------:|----------:|-------:|---------:|
| Decision Tree | 0.728    | 0.264     | 0.255  | 0.259    |
| Random Forest | 0.816    | 1.000     | 0.018  | 0.036    |

*(Full classification reports and confusion matrices are in
`outputs/`.)*

**Top features driving attrition (Random Forest importance):**
`MonthlyIncome`, `HourlyRate`, `MonthlyRate`, `DailyRate`,
`DistanceFromHome`, `TotalWorkingYears`, `Age`, `PercentSalaryHike`,
`NumCompaniesWorked`, `YearsAtCompany`.

Charts: `outputs/confusion_matrices.png`, `outputs/feature_importance_rf.png`,
`outputs/metric_comparison.png`.

## Model Comparison — Observations
1. **Random Forest wins on accuracy** (0.816 vs 0.728) and is far more
   conservative on the minority ("Yes") class — every positive
   prediction it makes is correct (precision = 1.00), but it only
   catches ~2% of employees who actually leave (recall = 0.018).
2. **Decision Tree is more balanced but noisier** — it flags more
   at-risk employees (recall = 0.255) but with more false alarms
   (precision = 0.264), and its single-tree structure makes it more
   sensitive to how the training data happened to split.
3. **Class imbalance dominates both models.** Only ~19% of employees
   in this data left the company, so a model that just predicts "No"
   most of the time gets rewarded with high accuracy while missing
   the group the business actually cares about (leavers). This is why
   precision/recall/F1 on the minority class matter more here than
   accuracy alone.
4. **Random Forest gives more stable, trustworthy feature importances**
   because they are averaged over 100 trees instead of coming from
   one tree's specific splits, which is useful for HR to act on
   (e.g., income, commute distance, and tenure show up as top
   drivers).

## Conclusion
On this dataset, **Random Forest produced the higher overall accuracy
(0.816 vs 0.728)**, but neither model is actually "better" once
minority-class recall is considered — the Decision Tree caught more
true leavers, while Random Forest was overly cautious and almost
never predicted attrition. Random Forest generally outperforms a
single Decision Tree because it builds many trees on bootstrapped
samples and random feature subsets, then averages their votes; this
ensembling reduces the variance and overfitting that a single tree is
prone to, and it also yields more reliable feature-importance
estimates. A key limitation of Decision Trees is that they overfit
easily and are unstable — a small change in the training data can
produce a very different tree. A key limitation of Random Forests is
reduced interpretability and higher computational/training cost, and,
as seen here, an ensemble can still be skewed toward the majority
class on imbalanced data unless techniques like class weighting or
resampling (e.g., SMOTE) are applied. In a real deployment, the
model choice should be paired with imbalance-handling and tuned
around business-relevant metrics (recall on leavers), not accuracy
alone.
