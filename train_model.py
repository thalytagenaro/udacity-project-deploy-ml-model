'''
Script to train machine learning model

Author : Thalyta
Date : March 2024
'''

import pandas as pd
import logging
import joblib


from sklearn.model_selection import train_test_split
from ml import (
    process_data,
    train_model,
    compute_model_metrics,
    inference,
    compute_slices
)

logging.basicConfig(
    filename='logs/train_model.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

logging.info("Reading cleaned data")
data = pd.read_csv(
    "./data/census.csv"
)

logging.info("Spliting train and test datasets")
train, test = train_test_split(data, test_size=0.20)

logging.info("Defining categorical features")
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

logging.info("Processing data")
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=categorical_features,
    label="salary",
    training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=categorical_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

logging.info("Training model")
model = train_model(
    X_train,
    y_train
)

logging.info("Saving model files")
model_files = {
    "model.pkl": model,
    "encoder.pkl": encoder,
    "lb.pkl": lb,
}
for filename, content in model_files.items():
    joblib.dump(
        content,
        f"./model/{filename}"
    )

logging.info("Evaluating model")
preds = inference(
    model=model,
    X=X_test
)
precision, recall, fbeta = compute_model_metrics(
    y=y_test,
    preds=preds
)
logging.info(
    f"Model performances: precision={precision:.2f}, recall={recall:.2f}, fbeta={fbeta:.2f}"
)

logging.info("Slicing data")
compute_slices(
    data=test,
    categorical_features=categorical_features,
    y=y_test,
    y_pred=preds,
)