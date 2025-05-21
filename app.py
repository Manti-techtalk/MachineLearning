from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

app = FastAPI()

model = load('model.joblib')

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post('/predict')
def predict(data: IrisData):
    data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(data)
    return {'prediction': int(prediction[0])}

@app.get("/")
def root():
    return {"message": "FastAPI is working"}
