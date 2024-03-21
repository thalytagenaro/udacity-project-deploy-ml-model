'''
Test model functions in 'ml' module

Author : Thalyta
Date : March 2024
'''

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import logging

from sklearn.ensemble import RandomForestClassifier
from ml import (
    train_model,
    compute_model_metrics,
    inference
)

logging.basicConfig(
    filename='./logs/test_model.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def test_data():
    '''
    Tests if data can be read
    '''
    try:
        data = pd.read_csv("./data/census.csv")
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        logging.info("Testing if data can be read - SUCCESS: Input path works")
    except FileNotFoundError:
        logging.error(
            "Testing if data can be read - ERROR: Input path does not work")


def test_train_model():
    '''
    Tests train_model() function from ml.model
    '''
    try:
        X = np.random.rand(100, 10)
        y = np.random.randint(2, size=100)
        assert isinstance(
            train_model(
                X_train=X,
                y_train=y
            ),
        RandomForestClassifier
        )
        logging.info(
            "Testing test_train_model() - SUCCESS: train_model() function from ml.model runs effectively")
    except Exception:
        logging.info(
            "Testing test_train_model() - ERROR: train_model() function from ml.model failed to run")


def test_compute_model_metrics():
    '''
    Tests compute_model_metrics() function from ml.model
    '''
    try:
        y = [1, 1, 0]
        y_pred = [1, 0, 0]
        precision, recall, fbeta = compute_model_metrics(
            y=y,
            preds=y_pred,
        )
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(fbeta, float)
        logging.info(
            "Testing test_compute_model_metrics() - SUCCESS: compute_model_metrics() function from ml.model runs effectively")
    except Exception:
        logging.info(
            "Testing test_compute_model_metrics() - ERROR: compute_model_metrics() function from ml.model failed to run")


def test_inference():
    '''
    Tests inference() function from ml.model
    '''
    try:
        X = np.random.rand(100, 10)
        y = np.random.randint(2, size=100)
        model = train_model(X, y)
        assert isinstance(
            inference(model, X),
            np.ndarray
        )
        logging.info(
            "Testing test_inference() - SUCCESS: inference() function from ml.model runs effectively")
    except Exception:
        logging.info(
            "Testing test_inference() - ERROR: inference() function from ml.model failed to run")


if __name__ == "__main__":
    test_data()
    test_train_model()
    test_compute_model_metrics()
    test_inference()