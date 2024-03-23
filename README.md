# Udacity Project: Deploy Machine Learning Model

This repository contains the code and related files for deploying a machine learning model. This project is part of the Machine Learning DevOps Engineer Nanodegree.

## Installation

To run the application locally, follow these steps:

1. Clone the repository to your local machine:

    ```
    git clone https://github.com/thalytagenaro/udacity-project-deploy-ml-model.git
    ```

2. Navigate to the project directory:

    ```
    cd udacity-project-deploy-ml-model
    ```

3. Install the required dependencies with a conda virtual environment:

    ```
    conda create -n udacity-deploy-api python=3.8
    conda activate udacity-deploy-api
    pip install -r requirements.txt
    ```

## Usage

Once you have installed the dependencies, you can train the model with:

```
python train_model.py
```

And run the API by using:
```
python main.py
```