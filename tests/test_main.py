'''
Test application built in main.py 

Author : Thalyta
Date : March 2024
'''

import sys
sys.path.append('.')

import logging

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    '''
    Tests application "root"
    '''
    try:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "Welcome!"
        logging.info("Testing application welcome page - SUCCESS: achieved expected response")
    except Exception:
        logging.error(
            "Testing application welcome page - ERROR: unexpected response")
        

def test_prediction_case_1():
    '''
    Tests application "predict" for response = "<=50K"
    '''
    try:
        response = client.post('/predict', json={
            "age": 38,
            "workclass": "Private",
            "fnlgt": 215646,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Divorced",
            "occupation": "Handlers-cleaners",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
            }
        )
        assert response.status_code == 200
        assert response.json() ==  "<=50K"
        logging.info("Testing application prediction <=50K - SUCCESS: achieved unexpected response")
    except Exception:
        logging.error(
            "Testing application prediction <=50K - ERROR: unexpected response")

    
def test_prediction_case_2():
    '''
    Tests application "predict" for response = ">50K"
    '''
    try:
        response = client.post('/predict', json={
            "age": 52,
            "workclass": "Self-emp-inc",
            "fnlgt": 287927,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital-gain": 15024,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
            }
        )
        assert response.status_code == 200
        assert response.json() ==  ">50K"
        logging.info("Testing application prediction >50K - SUCCESS: achieved unexpected response")
    except Exception:
        logging.error(
            "Testing application prediction >50K - ERROR: unexpected response")


if '__name__' == '__main__':
    test_root()
    test_prediction_case_1()
    test_prediction_case_2()