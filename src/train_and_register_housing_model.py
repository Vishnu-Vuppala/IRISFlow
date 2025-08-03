from math import sqrt
from pathlib import Path
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

experiment_id = mlflow.set_experiment("Housing Model Experiment")
print("Experiment ID:", experiment_id)
print("Tracking URI:", mlflow.get_tracking_uri())

housing = pd.read_csv("data/housing/housing.csv")
X = housing.drop("target", axis=1)
y = housing["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5),
}

client = MlflowClient()
best_rmse = float("inf")
best_model_info = None

for name, model in models.items():
    with mlflow.start_run(run_name=f"Housing-{name}"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.set_tag("type", "regression")
        mlflow.log_param("model", name)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"rmse": rmse, "r2_score": r2})

        signature = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(
            model, "model", signature=signature, input_example=X_test[:1]
        )

        print(f"[Housing] {name} RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_info = {
                "name": f"Housing-{name}",
                "run_id": mlflow.active_run().info.run_id,
            }

if best_model_info:
    model_uri = f"runs:/{best_model_info['run_id']}/model"
    registered_model_name = best_model_info["name"]
    try:
        mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        print(f"✅ Best housing model '{registered_model_name}' registered.")
    except Exception as e:
        print(f"⚠️ Model registration failed: {e}")
local_model_path = "housing_model"
Path(local_model_path).mkdir(exist_ok=True)
mlflow.sklearn.save_model(model, path=local_model_path)
print(f"✅ Best model also saved locally to '{local_model_path}/'")
