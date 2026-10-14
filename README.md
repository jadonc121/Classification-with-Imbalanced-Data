# Binary Classification with Imbalanced Data

_**Early project (2023).** I wrote this as one long notebook before moving into industry. I've kept it to show my starting point and left the analysis as-is. I now structure projects as modular, reusable Python code. See [my GitHub profile](https://github.com/jadonc121) for what that looks like._

Comparing resampling techniques and classifiers on imbalanced credit-risk data, with cross-validation designed to avoid data leakage.

**Notebook:** [Binary Classification with Imbalanced Data and Cross-Validation (Final Version).ipynb](Binary%20Classification%20with%20Imbalanced%20Data%20and%20Cross-Validation%20(Final%20Version).ipynb)

## Problem
When one class is much rarer than the other (fraud, disease, default), classifiers tend to favor the majority class and miss the cases that matter. This project tests how well different resampling strategies correct for that.

## Data
German bank credit data: 1,000 clients, 20 features (numeric and categorical), labeled low risk (70%) or high risk (30%). The goal is to avoid **false positives**, meaning a high-risk client classified as low risk, so **precision** on the low-risk class is the primary metric.

## Approach
- **Preparation:** one-hot encoded 13 categorical features (48 features total), stratified 80/20 train/holdout split.
- **Samplers (3):** random undersampling, random oversampling, SMOTE.
- **Models (4):** logistic regression, bagged decision trees, gradient boosting, random forest.
- **Grid:** all 12 sampler x model combinations, evaluated with 10-fold stratified cross-validation.
- **Leakage-safe design:** resampling and scaling are applied **inside** each training fold, never before the split. Resampling first would put duplicated or synthetic points into the validation folds and inflate scores.
- **Metrics:** precision, F1, ROC-AUC, accuracy, and run time, shown as heatmaps across all 12 combinations.
- **Reusable function:** a single `binary_model(...)` function runs any sampler/model combination through the same pipeline.

## Results
Precision (mean of 10 folds), selected combinations:

| Sampler | Logistic Reg. | Bagging | Gradient Boosting | Random Forest |
|---|---|---|---|---|
| Random undersampling | 0.852 | **0.875** | 0.845 | 0.870 |
| Random oversampling | 0.856 | 0.855 | 0.817 | 0.854 |
| SMOTE | 0.805 | 0.820 | 0.819 | 0.823 |

- **Ensembles with random undersampling gave the highest precision**; bagging + undersampling was selected (precision 0.875, AUC 0.707 in the grid).
- **SMOTE had the lowest precision but the highest F1 and accuracy**, which is the usual precision/recall tradeoff. Accuracy was treated as the least useful metric on imbalanced data.
- Re-running the selected combination on the full dataset with 10-fold CV: **precision 0.864, F1 0.741, ROC-AUC 0.704, accuracy 0.683**.

## Notes and future work
- Feature engineering (polynomial features + PCA) was tried and dropped because it lowered performance across all combinations (documented in the notebook appendix).
- Possible extensions: an unsampled baseline for comparison, and feature selection without feature engineering.

## Skills demonstrated
Imbalanced classification, resampling (SMOTE, over/undersampling), cross-validation without leakage, ensemble methods, model evaluation and metric selection, scikit-learn, imbalanced-learn.

## How to run
Python 3 with `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`. Place the dataset file (`german.csv`) next to the notebook and run top to bottom.
