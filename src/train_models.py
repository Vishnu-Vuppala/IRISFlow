import pandas as pd
import mlflow
import numpy as np
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Load datasets
housing = pd.read_csv("data/housing/housing.csv")
iris = pd.read_csv("data/iris/iris.csv")

def train_housing_models():
    X = housing.drop("target", axis=1)
    y = housing["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor()
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Housing-{name}"):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)

            mlflow.log_param("model", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(model, name="model", input_example=X_test[:1])
            print(f"[Housing] {name} RMSE: {rmse:.4f}")

def train_iris_models():
    X = iris.drop("target", axis=1)
    y = iris["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForestClassifier": RandomForestClassifier()
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Iris-{name}"):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)

            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, name="model", input_example=X_test[:1])
            print(f"[Iris] {name} Accuracy: {acc:.4f}")

if __name__ == "__main__":
    mlflow.set_experiment("MLflow-Demo")
    train_housing_models()
    train_iris_models()