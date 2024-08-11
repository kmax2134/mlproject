import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import category_encoders as ce
from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# Custom exception handling
class CustomException(Exception):
    def __init__(self, message, sys_module):
        super().__init__(message)
        self.sys_module = sys_module

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the data transformation pipeline
        '''
        try:
            numerical_columns = ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']
            ordinal_columns = ['property_type', 'furnishing_type', 'luxury_category', 'floor_category']
            onehot_columns = ['agePossession']
            target_columns = ['sector']

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline with multiple encoders
            cat_pipeline = ColumnTransformer(
                transformers=[
                    ("ordinal", Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal_encoder", OrdinalEncoder())
                    ]), ordinal_columns),

                    ("onehot", Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder", OneHotEncoder(drop='first', sparse_output=False)),
                        ("scaler", StandardScaler(with_mean=False))
                    ]), onehot_columns),

                    # TargetEncoder is handled separately
                ],
                remainder='passthrough'
            )

            logging.info(f"Categorical columns: {ordinal_columns + onehot_columns + target_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, ordinal_columns + onehot_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
  

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function reads the data, applies the transformations, and returns the transformed data
        '''
        try:
            # Reading the training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Defining target and numerical columns
            target_column_name = "price"  # Assuming 'price' is your target column
            numerical_columns = ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']

            # Splitting data into input features and target feature
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Manually applying target encoding to the 'sector' column
            target_encoder = ce.TargetEncoder(cols=['sector'])
            input_feature_train_df['sector'] = target_encoder.fit_transform(input_feature_train_df['sector'], target_feature_train_df)
            input_feature_test_df['sector'] = target_encoder.transform(input_feature_test_df['sector'])

            # Obtaining the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Applying transformations to the input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combining input features with the target feature for both train and test sets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # Saving the preprocessing object for later use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
