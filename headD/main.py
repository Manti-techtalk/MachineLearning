from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import numpy as np

app = FastAPI()
model = load('heart.joblib')

class HeartDiseaseData(BaseModel):
    age: float
    ca:float
    chol:float

@app.post('/predict')
def predict_heart_disease(data:HeartDiseaseData):
    input_data = np.array([data.age,data.ca,data.chol])
    input_data = input_data.reshape(1,-1)
    prediction = model.predict(input_data)
    return {'prediction': int(prediction[0])}
