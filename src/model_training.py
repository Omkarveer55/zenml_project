import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Union

from abc import ABC,abstractmethod


class model(ABC):
    @abstractmethod
    def training(self,X_train,y_train):
        pass

class logisticregression(model):
    def training(self, X_train, y_train, **kwargs):
        try:
            lg_model = LogisticRegression()
            lg_model.fit(X_train,y_train)
            logging.info('Model training done for logistic regression')
            return lg_model
        except Exception as e:
            logging.error(f'Error while training : {e}')
            raise e
        

