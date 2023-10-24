import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union

from abc import ABC,abstractmethod


class datastratergy(ABC):
    @abstractmethod
    def handling(self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class data_preprocessing(datastratergy):
    def handling(self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            return data
        except Exception as e:
            logging.error(f'Error while preprocessing data : {e}')
            raise e

class data_divide(datastratergy):

    def handling(self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            X = data.drop('target',axis=1)
            y = data['target']
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error(f'Error while splitting the data : {e}')
            raise e

class cleaning_data:
    def __init__(self,data:pd.DataFrame,stratergy:datastratergy):
        self.data = data
        self.stratergy = stratergy

    def handle_data(self):
        try:
            return self.stratergy.handling(self.data)
        except Exception as e:
            logging.error(f'Error while handling data : {e}')
            raise e
        

