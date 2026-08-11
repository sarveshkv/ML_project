import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
import dill

def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            model_params = params[model_name]

            gs = GridSearchCV(
                model,
                model_params,
                cv=3
            )

            # Train model
            gs.fit(X_train, y_train)

            # Set best parameters
            model.set_params(**gs.best_params_)

            # Train model with best parameters
            model.fit(X_train, y_train)

            # Predict testing data
            y_test_pred = model.predict(X_test)

            # Get R2 score
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)