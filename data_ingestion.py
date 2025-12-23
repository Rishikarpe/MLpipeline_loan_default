import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_data("data/credit_risk_dataset.csv")
    print(df.head())
    print(df.info())
