import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
def save_object(file_path, obj):
    import pickle
    with open(file_path, 'wb') as file_obj:
        pickle.dump(obj, file_obj)