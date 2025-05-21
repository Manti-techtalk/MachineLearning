from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load 
import numpy as np


app = FastAPI()

class CancerData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
@app.post('/predict')
def predict_cancer(data:CancerData):
    data = np.array([[data.mean_radius, data.mean_texture, data.mean_perimeter, data.mean_area, data.mean_smoothness,
                      data.mean_compactness, data.mean_concavity, data.mean_concave_points, data.mean_symmetry,
                      data.mean_fractal_dimension, data.radius_error, data.texture_error, data.perimeter_error,
                      data.area_error, data.smoothness_error, data.compactness_error, data.concavity_error,
                      data.concave_points_error, data.symmetry_error, data.fractal_dimension_error]])
    prediction = model.predict(data)
    return {'prediction': int(prediction[0])}

