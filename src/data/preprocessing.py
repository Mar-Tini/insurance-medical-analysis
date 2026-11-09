import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# -- Split data --

def split_data(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    os.makedirs("data/processed", exist_ok=True)

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    return train_df, test_df


# -- Preprocessing --

def preprocessing_data(df):

    df = df.drop(columns=["region", "children"])

    continuous = ["age", "bmi"]
    categorical = ["sex", "smoker"]

    train_df, test_df = split_data(df)

    X_train = train_df.drop(columns=["charges"])
    y_train = train_df["charges"]

    X_test = test_df.drop(columns=["charges"])
    y_test = test_df["charges"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), continuous),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    return X_train, X_test, y_train, y_test