import mlflow.pyfunc
deps = mlflow.pyfunc.get_model_dependencies("housing_model")
print(deps)