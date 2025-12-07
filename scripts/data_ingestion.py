from scripts.constants import *
import pandas as pd
from helpers.config import load_config
import pandas as pd

from helpers.logger import logger

class DataIngestion:
    """
    A utility class for retrieving data from FredAPI.
    """
    def __init__(self,config: dict):
        """"""
        self.config = config or load_config()
        
    def fetch_fred_data(self) -> pd.DataFrame:
        """
        Fetch data from FredAPI.
        
        Returns:
            data (pd.Dataframe): a dataframe consisting of Fred FED FUNDS series data.
        """
        # RAW PATH where the data will be stored
        RAW_PATH = self.config['raw_path']
        
        # FED FUNDS from fred
        FEDFUNDS = fred.get_series("FEDFUNDS")
     
        FEDFUNDS.name = "FEDFUNDS"
        
        data = pd.DataFrame(FEDFUNDS).dropna()
        
        data = data.reset_index()

        data['date'] = data["index"]
        data.drop("index",inplace=True,axis=1)
        data.drop_duplicates(inplace=True)    
    
    
        return data
    
    

    
    
    
            
        
        




