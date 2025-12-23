import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("data/cleaned_credit_risk_dataset.csv")

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# MLflow Experiment
# -----------------------------
mlflow.set_experiment("loan_default_xgboost")

with mlflow.start_run():

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # -----------------------------
    # Log to MLflow
    # -----------------------------
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)

    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.xgboost.log_model(model, artifact_path="model")

    print(f"ROC-AUC: {roc_auc:.4f}")
