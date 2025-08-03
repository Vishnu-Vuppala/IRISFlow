from pathlib import Path
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

mlflow.set_experiment("Iris Model Experiment")

iris = pd.read_csv("data/iris/iris.csv")
X = iris.drop("target", axis=1)
y = iris["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100)
}

client = MlflowClient()
best_acc = 0
best_model_info = None

for name, model in models.items():
    with mlflow.start_run(run_name=f"Iris-{name}"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.set_tag("type", "classification")
        mlflow.log_param("model", name)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        signature = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_test[:1])

        print(f"[Iris] {name} Accuracy: {acc:.4f}")

        if acc >= best_acc:
            best_acc = acc
            best_model_info = {
                "name": f"Iris-{name}",
                "run_id": mlflow.active_run().info.run_id
            }

# if best_model_info:
#     model_uri = f"runs:/{best_model_info['run_id']}/model"
#     registered_model_name = best_model_info["name"]
#     try:
#         client.create_registered_model(registered_model_name)
#     except:
#         pass
#     client.create_model_version(name=registered_model_name, source=model_uri, run_id=best_model_info["run_id"])
#     print(f"✅ Best iris model '{registered_model_name}' registered.")

if best_model_info:
    model_uri = f"runs:/{best_model_info['run_id']}/model"
    registered_model_name = best_model_info["name"]
    
    try:
        mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        print(f"✅ Best Iris model '{registered_model_name}' registered.")
    except Exception as e:
        print(f"⚠️ Model Iris failed: {e}")

# Save model locally so Docker container can use it
local_model_path = "iris_model"
Path(local_model_path).mkdir(exist_ok=True)
mlflow.sklearn.save_model(model, path=local_model_path)
print(f"✅ Best model also saved locally to '{local_model_path}/'")