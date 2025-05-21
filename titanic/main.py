from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

app = FastAPI()

model = load('titanic_model.joblib')

class TitanicData(BaseModel):
    Pclass: int
    PassengerId: int
    Age: float

@app.post('/predict')
def predict_titanic(data: TitanicData):
    input_data = [[data.Pclass, data.PassengerId, data.Age]]
    prediction = model.predict(input_data)
    return {'prediction': int(prediction[0])}
# To run the FastAPI server, use the command:
# uvicorn main:app --reload