import pandas as pd
import numpy as np
import logging

from zenml import step

class ingesting_data:
    def __init__(self,path : str):
        self.path = path
    
    def get_data(self):
        return pd.read_csv(self.path)
    

@step
def ingest_data(path:str) -> pd.DataFrame:
    try:
        data_ingestion = ingesting_data(path)
        data = data_ingestion.get_data()
        return data
    except Exception as e:
        logging.error(f"Error in ingesting the data : {e}")
        raise e