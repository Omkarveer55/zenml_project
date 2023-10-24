import logging
import pandas as pd
import numpy as np
from src.model_training import logisticregression
from sklearn.base import ClassifierMixin
from config import modelname
import mlflow

from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def training(X_train : pd.DataFrame,
             y_train:pd.Series,
             X_test:pd.DataFrame,
             y_test:pd.Series,
             config:modelname) -> ClassifierMixin:
    
    try:
        model = None
        if config.model_name == "LogisticRegression":
            mlflow.sklearn.autolog()
            model = logisticregression()
            trained_model = model.training(X_train=X_train,y_train=y_train)
            logging.info('model training finished')
            return trained_model
        else:
            raise ValueError(f'model {config.model_name} is not supported')
    except Exception as e:
        logging.error(f'error while training : {e}')
        raise e
    
