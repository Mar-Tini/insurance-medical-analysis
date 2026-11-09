import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -- Interaction features --
    df["age_bmi"] = df["age"] * df["bmi"]
    df["smoker_bmi"] = df["smoker"].map({"yes": 1, "no": 0}) * df["bmi"]
    df["smoker_age"] = df["smoker"].map({"yes": 1, "no": 0}) * df["age"]

    # -- Polynomial features --
    df["age_squared"] = df["age"] ** 2
    df["bmi_squared"] = df["bmi"] ** 2

    # -- Group features --
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 40, 55, np.inf],
        labels=["young", "adult", "middle_age", "senior"]
    )

    df["bmi_group"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["underweight", "normal", "overweight", "obese"],
        right=False
    )

    return df