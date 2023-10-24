import logging
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from src.model_evaluation import r2score,mse
from typing import Tuple
from typing_extensions import Annotated
import mlflow

from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate(model : ClassifierMixin,
            y_test:pd.Series,
            X_test:pd.DataFrame) -> Tuple[
                Annotated[float,"r2_score"],
                Annotated[float,"mse"]
            ]:
    try:
        y_pred = model.predict(X_test)
        cal_r2score = r2score()
        r2_score_cal = cal_r2score.calculate_score(y_actual=y_test,y_pred=y_pred)
        mlflow.log_metric("r2_score",r2_score_cal)
        mse_score = mse()
        score_mse = mse_score.calculate_score(y_actual=y_test,y_pred=y_pred)
        mlflow.log_metric("mse",score_mse)
        return r2_score_cal,score_mse
    except Exception as e:
        logging.error(f'error while evaluation : {e}')
        raise e