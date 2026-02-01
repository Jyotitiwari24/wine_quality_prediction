import pandas as pd


def load_data(file_path):
    """Load CSV data."""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Preprocess wine dataset: create binary label for quality."""
    X = df.drop('quality', axis=1)
    Y = df['quality'].apply(lambda y: 1 if y >= 7 else 0)  # 1: Good, 0: Bad
    return X, Y
