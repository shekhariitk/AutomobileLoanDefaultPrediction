import os
import sys
import pickle
import pandas as pd
import numpy as np
import yaml

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e: 
        raise CustomException(e, sys)
    
# Load YAML configuration
def load_config(config_file_path):
    try:
        with open(config_file_path, 'r') as file:
          return yaml.safe_load(file)
        
    except Exception as e: 
        logging.info("Error Occured during load_config ")
        raise CustomException(e, sys)