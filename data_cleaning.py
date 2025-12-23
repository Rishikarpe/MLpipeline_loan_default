import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # Handle Missing Values
    # -----------------------------
    df["person_emp_length"] = df["person_emp_length"].fillna(
        df["person_emp_length"].median()
    )

    df["loan_int_rate"] = df["loan_int_rate"].fillna(
        df["loan_int_rate"].median()
    )


    # -----------------------------
    # Binary Encoding
    # -----------------------------
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map(
        {"Y": 1, "N": 0}
    )

    # -----------------------------
    # Ordinal Encoding (loan_grade)
    # -----------------------------
    grade_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    df["loan_grade"] = df["loan_grade"].map(grade_mapping)

    # -----------------------------
    # One-Hot Encoding
    # -----------------------------
    categorical_cols = ["person_home_ownership", "loan_intent"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/credit_risk_dataset.csv")
    cleaned_df = clean_data(df)

    # Save cleaned dataset
    cleaned_df.to_csv("data/cleaned_credit_risk_dataset.csv", index=False)
    print("Cleaned dataset saved to data/cleaned_credit_risk_dataset.csv")

    print(cleaned_df.head())
    print(cleaned_df.info())
