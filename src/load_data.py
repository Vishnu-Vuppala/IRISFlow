from sklearn.datasets import (
    fetch_california_housing,
    load_iris as sklearn_load_iris,
)

import pandas as pd
import os


def save_dataframe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


def load_housing_data():
    housing = fetch_california_housing(as_frame=True)
    df = pd.concat([housing.data, housing.target.rename("target")], axis=1)
    # Preprocess: Convert all int columns to float
    int_columns = df.select_dtypes(include="int").columns
    df = df.astype({col: "float64" for col in int_columns})
    # Example: Rename columns to be MLflow-friendly (no spaces, parens)
    df.columns = df.columns.str.replace(r"[()\s]", "_", regex=True)
    # Save
    save_dataframe(df, "data/housing/housing.csv")


def load_iris_data():
    iris = sklearn_load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    # Preprocess
    int_columns = df.select_dtypes(include="int").columns
    df = df.astype({col: "float64" for col in int_columns})

    df.columns = df.columns.str.replace(r"[()\s]", "_", regex=True)
    save_dataframe(df, "data/iris/iris.csv")


if __name__ == "__main__":
    load_housing_data()
    load_iris_data()
