from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Housing Model API")

# Load the registered model
model_name = "Housing-DecisionTreeRegressor"  # Use your registered model name here
#model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/1")
model = mlflow.pyfunc.load_model("housing_model")
# Define input schema
class HousingInput(BaseModel):
    MedInc: float   
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(data: HousingInput):
    try:
        # Convert input to DataFrame with one row
        input_df = pd.DataFrame([data.dict()])
        
        # Get prediction from MLflow model
        prediction = model.predict(input_df)
        
        # prediction can be numpy array or list; return first value
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
