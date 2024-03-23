'''
Script to train machine learning model

Author : Thalyta
Date : March 2024
'''

import uvicorn
import pandas as pd
import joblib

from pydantic import BaseModel, Field
from fastapi import FastAPI
from ml import process_data, inference

# Loading model
model = joblib.load('model/model.pkl')
encoder = joblib.load('model/encoder.pkl')
lb = joblib.load('model/lb.pkl')

# Defining categorical features
categorical_features = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]

# Defining input data (sample)
class InputData(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example='State-gov')
    fnlgt: int = Field(example=77516)
    education: str = Field(example='Bachelors')
    education_num: int = Field(alias='education-num', example=13)
    marital_status: str = Field(alias='marital-status', example='Never-married')
    occupation: str = Field(example='Adm-clerical')
    relationship: str = Field(example='Not-in-family')
    race: str = Field(example='White')
    sex: str = Field(example='Male')
    capital_gain: int = Field(alias='capital-gain', example=2174)
    capital_loss: int = Field(alias='capital-loss', example=0)
    hours_per_week: int = Field(alias='hours-per-week', example=40)
    native_country: str = Field(alias='native-country', example='United-States')


# Initializing API object
app = FastAPI()


# # Creating App
@app.get('/')
async def root():
    return {'message': 'Welcome!'}


@app.post('/predict')
async def predict(input_data: InputData):
    data = pd.DataFrame(
        {column: value for column, value in input_data.dict(by_alias=True).items()},
        index=[0]
    )
    X, * _ = process_data(
        data,
        categorical_features=categorical_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    prediction = inference(model, X)
    return lb.inverse_transform(prediction)[0]


if __name__ == "__main__":
    uvicorn.run('main:app', host='0.0.0.0', port=5000, reload=True)