# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model uses the classifier Random Forest from `sklearn.ensemble.RandomForestClassifier`.

## Intended Use

This model predicts the category of the salary of an individual (if it is over or less than 50k). 

## Training Data

The training data is the Census Income from the UCI library, that is available in [this link](https://archive.ics.uci.edu/dataset/20/census+income).

## Evaluation Data

The dataset is divided into training (80%) and testing (20%) subsets, adhering to the 80/20 ratio. Only categorical features are considered for preprocessing and the data cleaning procedure can be found in the notebook within `notebooks/clean_data.ipynb`.

## Metrics
As it can be seen in `logs/train_model.log`:
- Precision: 0.72
- Recall: 0.63
- F-beta: 0.67

## Ethical Considerations

- The Census Income dataset has been rendered anonymous;
- The model is impartial towards any specific demographic group.

## Caveats and Recommendations

- Enhance feature selection and engineering methods to capture relevant information and improve model interpretability;
- Update the dataset to incorporate new insights and ensure model adaptability to changing environments; and
- Hyperparameter tuning techniques to optimize model performance and enhance predictive accuracy.