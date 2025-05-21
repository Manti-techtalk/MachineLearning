from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import numpy as np

app = FastAPI()
model = load('insurance_model.joblib')

class InsuranceData(BaseModel):
    age: float
    bmi: float
    region: str

@app.post('/predict')
def predict_insurance(data: InsuranceData):
    # Convert region to one-hot encoding
    region = data.region
    if region == 'southeast':
        region_encoded = [1, 0, 0]
    elif region == 'southwest':
        region_encoded = [0, 1, 0]
    elif region == 'northeast':
        region_encoded = [0, 0, 1]
    else:
        region_encoded = [0, 0, 0]
    input_data = np.array([[data.bmi, data.age] + region_encoded])
    prediction = model.predict(input_data)
    return {'prediction': int(prediction[0])}

