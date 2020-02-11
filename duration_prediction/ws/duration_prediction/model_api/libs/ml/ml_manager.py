# -*- coding: utf-8 -*-
#Author: Andrea Urgolo @Space1
import sys
from os import path
import os.path
import pandas as pd
import random
import time

# from scipy.stats.stats import pearsonr
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn import svm, preprocessing
# from sklearn.linear_model import Ridge, LogisticRegression
# from sklearn.neural_network import MLPRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import KFold
from sklearn.externals import joblib
# from xgboost import XGBRegressor

import pickle

#import from parent dir
sys.path.append(path.dirname(path.dirname(path.abspath(__file__)))+'/io')
from data_manager import *

class MLManager:
    """ ML manager: machine learning functions and models management """
    
    # class vars
    # e.g. N_SAMPLE = 10 # sample of data to get while testing

    def __init__(self, config, db_adapter, data_manager):
        self.task = config['ML_TASK']
        self.config = config
        self.db_adapter = db_adapter
        self.data_manager = data_manager

    # Importa i dati
    def predict_duration(self, oc, rs, odl, res, app):
        """ 
            Returns the duration predictions for the followng given data:

             - oc: operative center id
             - rs: list of resources id for which a prediction must be made
             - work order data (infos used to retrieve the work order on which we are making our predictions)
                * res = resource id (idrisorsa) 
                * app = agenda apointment id (aageid)
                * odl = odl id (codiceodl)

        """

        # get odl row from testset
        row = self.data_manager.get_test_row(oc, odl, res, app)

        # X = updated row data with res profiles
        X = self.data_manager.update_row_with_res_profiles(row, oc, rs)

        # Load model related files
        model = self.get_model_pipeline_from_file(oc)

        self.data_manager.set_columns_type(X)      

        # predict the durations and compute their exp minus 1 value
        preds = DataManager.list_expm1_transform(model.predict(X).tolist())

        # cut prediction bigger than the outliers limit
        preds = [int(p) if p < self.config['DURATION_OUTLIERS_LIMIT'] else self.config['DURATION_OUTLIERS_LIMIT'] for p in preds ]

        return pd.DataFrame(zip(rs, preds), columns=['id', 'pred'])

    ### Getters ###

    def get_model_data(self, oc): # FROM DB
        """ 
            Given an operative center id oc, returns the related model data tuple (model, features, dummies) where
            
            - model is the machine learning model specifically built for the operative center oc;
            - features are the columns used to fit the model
            - dummies are the features columns plus the dummies variables used to fit the model

            The data is loaded from DB

        """
        model_data = self.db_adapter.get_model_by_oc(oc)

        model, features, dummies = (None, None, None)
        if model_data:
            model = model_data['model']
            features = model_data['features']
            dummies = model_data['dummies']
        return (model, features, dummies)

    def get_model_data_from_files(self, oc):
        """ 
            Given an operative center id oc, returns the related model data tuple (model, features, dummies) where
            
            - model is the machine learning model specifically built for the operative center oc;
            - features are the columns used to fit the model
            - dummies are the features columns plus the dummies variables used to fit the model

            The data is loaded from file

        """
        # Load model related files
        model_path = self.config['DATA_PATH'] + self.config['CUSTOMER_NAME'] + '/models/'

        features_file = model_path + self.task + '_' + str(oc) + '_features.txt'
        dummies_file = model_path + self.task + '_' + str(oc) + '_dummies.txt'
        model_file =  model_path + self.task + '_' + str(oc) + '.joblib'

        if os.path.isfile(features_file) and os.path.isfile(dummies_file) and os.path.isfile(model_file):
            model = joblib.load(model_file)
            features = open(features_file, 'r', encoding=self.config['DATA_ENCODING']).read().rstrip('\n').split(self.config['DATA_SEPARATOR'])
            dummies = open(dummies_file, 'r', encoding=self.config['DATA_ENCODING']).read().rstrip('\n').split(self.config['DATA_SEPARATOR'])
            return (model, features, dummies)
        return (None, None, None)

    def get_model_pipeline_from_file(self, oc):
        """ 
            Given an operative center id oc, returns the related model pipeline.
            The data is loaded from file

        """
        # Load model related files
        model_path = self.config['DATA_PATH'] + self.config['CUSTOMER_NAME'] + '/models/'

        model_file =  model_path + self.task + '_' + str(oc) + '_pipeline.joblib'

        if os.path.isfile(model_file):
            model = joblib.load(model_file)
            return model
        return None


    def save_all_models(self):
        """ For each operative center saves all model data to the DB """
        # get the operative centers list from the dataset
        opcs = self.data_manager.get_oc_ids()

        # iterate over the opcs list, and save the model data for each operative center
        for oc in opcs:
            model, features, dummies = self.get_model_data_from_files(oc)
            if model is not None:
                self.save_model(oc, model, features, dummies)


    def  save_model(self, oc, model, features, dummies):
        """ Saves a model data to the DB """

        # model_path = self.config['DATA_PATH'] + self.config['CUSTOMER_NAME'] + '/models/'
        # model_file =  model_path + self.task + '_' + str(oc) + '.joblib'

        model_data = {
            'oc' : oc,
            'model' : pickle.dumps(model),
            'model_name': type(model).__name__,
            'model_description': str(model),
            'features' : features,
            'dummies' : dummies,
            'timestamp' : time.time()
        }

        self.db_adapter.save_model(**model_data)
