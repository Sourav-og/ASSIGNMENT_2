# ============================================
# LOAN DEFAULT PREDICTION - FINSECURE
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --------------------------------------------
# SECTION 1: LOAD DATA & PROBLEM FORMULATION
# --------------------------------------------

# Load the dataset (make sure loan_data.csv is in the same folder)
data = pd.read_csv("loan_data.csv")

# Target variable: loan_paid_back (1 = paid back, 0 = default)
target_col = "loan_paid_back"

# Drop columns not to be used as features
drop_cols = [target_col]
if "id" in data.columns:
    drop_cols.append("id")

X = data.drop(columns=drop_cols)
y = data[target_col]

# --------------------------------------------
# SECTION 2: PREPROCESSING PIPELINE
# --------------------------------------------

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = [col for col in X.columns if col not in numeric_features]

# Numeric transformer: impute missing values + scale
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical transformer: impute missing values + one-hot encode
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine both in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# --------------------------------------------
# TRAIN–TEST SPLIT
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # keeps class balance
)

# --------------------------------------------
# SECTION 3: MODEL DEVELOPMENT (Logistic Regression)
# --------------------------------------------

# You can change this to any single supervised model (e.g., RandomForestClassifier)
model = LogisticRegression(max_iter=1000)

# Full pipeline: preprocessing + model
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train the model
clf.fit(X_train, y_train)

# Predict probabilities for the positive class (loan_paid_back = 1)
y_proba_test = clf.predict_proba(X_test)[:, 1]

# Overall AUC on the test set
overall_auc = roc_auc_score(y_test, y_proba_test)
print(f"Overall Test AUC: {overall_auc:.4f}")

# --------------------------------------------
# SECTION 4: SUBGROUP ANALYSIS (Fairness)
# --------------------------------------------

# Add predictions back to a copy of test data for analysis
test_df = X_test.copy()
test_df[target_col] = y_test.values
test_df["pred_proba"] = y_proba_test

# Helper function to compute AUC for each subgroup safely
def subgroup_auc(df, group_col, target_col, proba_col):
    """
    Computes AUC for each subgroup in `group_col`.
    Skips subgroups that have only one class in the target (AUC not defined).
    """
    results = []

    for val, sub_df in df.groupby(group_col):
        y_true = sub_df[target_col]
        y_score = sub_df[proba_col]

        # AUC requires both 0 and 1 in the target
        if len(np.unique(y_true)) < 2:
            auc_value = np.nan  # not defined
        else:
            auc_value = roc_auc_score(y_true, y_score)

        results.append({"group": val, "auc": auc_value, "count": len(sub_df)})

    results_df = pd.DataFrame(results).sort_values(by="auc", ascending=False)
    return results_df

# --------------------------------------------
# A) By education_level
# --------------------------------------------

if "education_level" in test_df.columns:
    edu_auc_df = subgroup_auc(test_df, "education_level", target_col, "pred_proba")
    print("\nAUC by education_level:")
    print(edu_auc_df)
else:
    print("\nColumn 'education_level' not found in data.")

# --------------------------------------------
# B) By loan_purpose (Top 3 & Bottom 3)
# --------------------------------------------

if "loan_purpose" in test_df.columns:
    purpose_auc_df = subgroup_auc(test_df, "loan_purpose", target_col, "pred_proba")

    print("\nAUC by loan_purpose (all):")
    print(purpose_auc_df)

    # Drop NaN AUC values before ranking top/bottom
    valid_purpose_auc_df = purpose_auc_df.dropna(subset=["auc"])

    # Top 3
    top3 = valid_purpose_auc_df.head(3)
    # Bottom 3
    bottom3 = valid_purpose_auc_df.tail(3)

    print("\nTop 3 loan_purpose by AUC:")
    print(top3)

    print("\nBottom 3 loan_purpose by AUC:")
    print(bottom3)
else:
    print("\nColumn 'loan_purpose' not found in data.")
