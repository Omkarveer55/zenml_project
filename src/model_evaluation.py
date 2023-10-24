import logging
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
from typing import Union

from abc import ABC,abstractmethod


class evaluation(ABC):
    @abstractmethod
    def calculate_score(self,y_actual,y_pred):
        pass

class r2score(evaluation):
    def calculate_score(self, y_actual, y_pred):
        try:
            return r2_score(y_actual,y_pred)
        except Exception as e:
            logging.error(f'Error while calvculating r2_score : {e}')
            raise e
        
class mse(evaluation):
      def calculate_score(self, y_actual, y_pred):
        try:
            return mean_squared_error(y_actual,y_pred)
        except Exception as e:
            logging.error(f'Error while calvculating r2_score : {e}')
            raise e