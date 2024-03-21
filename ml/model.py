import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from ml import process_data


def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_slices(data, categorical_features, y, y_pred):
    """ Generates model metrics for slices of data. """
    slices_dict = {}

    features_list = []
    categories_list = []
    precisions_list = []
    recalls_list = []
    fbetas_list = []
    
    for feature in categorical_features:
        categories = list(data[feature].unique())
        
        for category in categories:
            features_list.append(feature)
            categories_list.append(category)

            filter = data[feature]==category

            y_filter = y[filter]
            y_pred_filter = y_pred[filter]

            precision, recall, fbeta = compute_model_metrics(
                y[filter],
                y_pred[filter]
            )

            precisions_list.append(precision)
            recalls_list.append(recall)
            fbetas_list.append(fbeta)

    slices_dict["feature"] = features_list
    slices_dict["category"] = categories_list
    slices_dict["precision"] = precisions_list
    slices_dict["recall"] = recalls_list
    slices_dict["fbeta"] = fbetas_list

    slices = pd.DataFrame(
        slices_dict
    )

    slices.to_csv(
        "./data/slices.csv"
    )

    return slices