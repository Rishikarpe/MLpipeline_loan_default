import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Load model from MLflow
# -----------------------------
MODEL_URI = "models:/LoanDefaultModel/Production"
model = mlflow.xgboost.load_model(MODEL_URI)

app = FastAPI(title="Loan Default Prediction API")

# -----------------------------
# Input Schema
# -----------------------------
class LoanInput(BaseModel):
    person_age: int
    person_income: int
    person_emp_length: float
    loan_grade: int
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: int
    cb_person_cred_hist_length: int
    person_home_ownership_OTHER: int
    person_home_ownership_OWN: int
    person_home_ownership_RENT: int
    loan_intent_EDUCATION: int
    loan_intent_HOMEIMPROVEMENT: int
    loan_intent_MEDICAL: int
    loan_intent_PERSONAL: int
    loan_intent_VENTURE: int

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: LoanInput):
    df = pd.DataFrame([data.dict()])
    proba = model.predict_proba(df)[0][1]
    return {"default_probability": float(proba)}
