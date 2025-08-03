from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Iris Model API")

model_name = "Iris-RandomForestClassifier"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/1")

label_map = {0.0: "setosa", 1.0: "versicolor", 2.0: "virginica"}

# Match input names to training schema
class IrisInput(BaseModel):
    sepal_length__cm_: float
    sepal_width__cm_: float
    petal_length__cm_: float
    petal_width__cm_: float

@app.post("/predict")
def predict(data: IrisInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        species = label_map.get(float(prediction[0]), "Unknown")
        return {"prediction": species}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
