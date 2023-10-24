import logging
import pandas as pd
import numpy as np
from src.data_cleaning import cleaning_data,data_preprocessing,data_divide
from typing import Tuple
from typing_extensions import Annotated

from zenml import step


@step
def cleaning(data:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]
]:
    try:
        process_stratergy = data_preprocessing()
        clean_data = cleaning_data(data,process_stratergy)
        preprocessed_data = clean_data.handle_data()
        divide_stratergy = data_divide()
        clean_data = cleaning_data(preprocessed_data,divide_stratergy)
        X_train,X_test,y_train,y_test = clean_data.handle_data()
        logging.info('completed the cleaning process')
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(f'Error while cleaning data : {e}')
        raise e
    