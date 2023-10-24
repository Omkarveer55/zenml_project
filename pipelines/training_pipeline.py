import logging
import pandas as pd
import numpy as np

from zenml import step
from zenml import pipeline

from steps.ingest_data import ingesting_data,ingest_data
from steps.clean import cleaning
from steps.train import training
from steps.evaluate import evaluate

@pipeline
def train_pipeline(data_path:str):
    data = ingest_data(data_path)
    print(data)
    X_train,X_test,y_train,y_test = cleaning(data)
    model = training(X_train,y_train,X_test,y_test)
    r2_score,mse = evaluate(model,y_test,X_test)


