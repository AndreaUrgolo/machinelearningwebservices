# -*- coding: utf-8 -*-
#Author: Andrea Urgolo
import numpy as np
import os.path
import pandas as pd
import random
import gc

from scipy.sparse import csr_matrix

from sklearn import preprocessing

class DataManager:
    """ Dataset manager: contains all function to manipulate and query the dataset """

    N_SAMPLE = 10 # sample of data to get while testing

    def __init__(self, config, db_adapter):
        self.config = config
        self.db_adapter = db_adapter
        self.res_profiles = dict()

        ## load data from config files (see app_config.py)

        d = dict() # temp dictionary for config vars readed from external files
        config_files = config['CONFIG_FILES']
        for k in config_files:
            if os.path.isfile(config_files[k]):
                exec(open(config_files[k]).read(), d)

        # Check if config files contain all required vars definition
        if not 'columns_type' in d: # use a default value
            raise AttributeError(f'Error in {self.__class__.__name__}: columns_type is missing.')
        if not 'valid_columns' in d: # use a default value
            raise AttributeError(f'Error in {self.__class__.__name__}: valid_columns is missing.')
        if not 'resource_columns' in d: # use a default value
            raise AttributeError(f'Error in {self.__class__.__name__}: resource_columns is missing.')

        # set loaded vars from config files
        for k in d:
            setattr(self, k, d[k]) # because self[k] = d[k] doesn't work :-)

        ## END loading data from config files

        # Init empty dataset vars
        self.dataset=None
        # train e test qui sotto sostituiti da metodi getter
        # self.test=None 
        # self.train=None

    # Importa i dati
    def init_data(self):
        db_adapter = self.db_adapter
        columns_type = self.columns_type
        config = self.config

        # only test env
        # print('Reading the dataset')
        # self.dataset = self.read_csv(config['DATASET_FILE'])

        # Export data in hdf5 file
        # self.dataset.to_hdf(config['DATASET_FILE']+'.h5', 'test')
        # Load data from hdf5 file
        print('Reading the dataset from HDF5 file')
        self.dataset = pd.read_hdf(config['DATASET_FILE']+'.h5', 'test')

        # # in production env use this instead
        # if not db_adapter.is_present(db_adapter.config['dataset_source']):
        #     print("Reading the dataset from source CSV file")
        #     self.dataset = self.read_csv(config['DATASET_FILE'])
        #     db_adapter.save_dataframe(db_adapter.config['dataset_source'], self.dataset)
        # else:
        #     print("Reading the dataset from DB")
        #     self.dataset = db_adapter.get_dataframe(db_adapter.config['dataset_source'], list(self.columns_type.keys()))

        print('DONE')

        DataManager.set_columns_type_from_dict(self.dataset, self.columns_type) # columns_type from config file "columns_type_file"

        #DataManager.compute_target_class(self.dataset)

        print("Dataset shape: ", self.dataset.shape)
        print("Dataset info")
        self.dataset.info()

        
        # Divisione tra train e test qui sotto sostituita da metodi getter (get_train() e get_test())
        # per ridurre l'uso della memoria
        # # Divide dataset into train and test sets using the configured TRAIN_FRACTION
        # limit = int(len(self.dataset)*(config['TRAIN_FRACTION']))

        # self.train = self.dataset.iloc[0:limit]
        # self.test = self.dataset.iloc[limit:]

        # garbage collector
        gc.collect()


    def set_columns_type(self, data):
        """ Set the columns type of the 'data' dataframe exploiting the columns_type 
            dictionary from loaded from config file.
        """
        fields_dict = self.columns_type
        for field in fields_dict:
            if field in data.columns:
                data[field] = data[field].astype(fields_dict[field])

    @staticmethod
    def set_columns_type_from_dict(data, fields_dict):
        """ Set columns type as the above function (static version)"""
        for field in fields_dict:
            if field in data.columns:
                data[field] = data[field].astype(fields_dict[field])


    def read_csv(self, path, sep=None, encoding=None):
        """ Returns a dataframe from a given csv file specified in 'path'. """
        if sep is None:
            sep=self.config['DATA_SEPARATOR']
        if encoding is None:
            encoding=self.config['DATA_ENCODING']

        return pd.read_csv(path, sep=sep, header=0, encoding=encoding, engine='python')


    ### Getters ###

    def get_dataset(self):
        return self.dataset
    def  get_resource_columns(self):
        return self.resource_columns
    def get_test(self):
        return self.dataset.iloc[self.get_train_test_limit():]
    def get_train(self):
        return self.dataset.iloc[0:self.get_train_test_limit()]
    def get_train_test_limit(self):
        return int(len(self.dataset)*(self.config['TRAIN_FRACTION']))


    """ Domain specific functions """

    def get_dataset_columns(self):
        """ Returns the dataset columns """
        return self.dataset.columns

    def get_dummies(self, data, dummies):
        """ 
            Return the data updated with the proper dummies variables, filtered in order to contain 
            only the dummies columns and typed as defined in the columns_type config file 

        """
        data_dum = pd.get_dummies(data)
        res = pd.DataFrame(columns=dummies)
        res = res.append(data_dum).fillna(0)[dummies]
        DataManager.set_columns_type_from_dict(res, self.columns_type)
        return res

    def  get_sparse_dummies(self, data, dummies):
        """ 
            Returns the compressed sparse row matrix data updated with the proper dummies variables, filtered in order to contain 
            only the dummies columns and typed as defined in the columns_type config file 

        """
        return self.get_sparse_matrix(self.get_dummies(data, dummies)[dummies])


    def get_sparse_matrix(self, data):
        """ Returns the compressed sparse row matrix of the data """
        return csr_matrix(data)

    @staticmethod
    def get_mean_int_range(x):
        """ get mean int value from range x """
        vals=x.split('-')
        if len(vals) == 2:  # class with two range delimeters "n1-n2"
            return int((int(vals[0])+int(vals[1]))/2)
        else: # last class "n+"
            return int(x.split('+')[0])

    @staticmethod
    def get_min_int_range(x):
        """ get min int value from range x """
        vals=x.split('-')
        if len(vals) == 2:  # class with two range delimeters "n1-n2"
            return int(vals[0])
        else: # last class "n+"
            return int(x.split('+')[0])

    @staticmethod
    def get_max_int_range(x):
        """ get max int value from range x """
        vals=x.split('-')
        if len(vals) == 2:  # class with two range delimeters "n1-n2"
            return int(vals[1])
        else: # last class "n+"
            return int(x.split('+')[0])

    
    @staticmethod
    def list_expm1_transform(list):
        return np.expm1(list)
                
    # def target_log1p_transform(self):
    #     self.dataset[self.config['Y_COLUMN']] = [t if not np.isinf(t) else -1.0 for t in np.log1p(self.dataset[self.config['Y_COLUMN']]) ]
        
        
    # def target_expm1_transform(self, df):
    #     df[self.config['Y_COLUMN']] = [ t for t in np.expm1(df[self.config['Y_COLUMN']]) ]
    
    

    def update_row_with_res_profiles(self, row, oc, rs):
        """ Returns a dataset with a row duplicated for each resources in the 'rs' ids list.
            Each examples is updated with the correspondent resource profiles data
        """
        X = pd.DataFrame(columns=self.dataset.columns)
        self.set_columns_type(X)        
        for i in range(len(rs)):
            res_profile=self.res_profiles[oc][rs[i]]

            # update profile data
            for x in res_profile.columns:
                row[x] = res_profile.iloc[0][x]
            X=X.append(row)

        return X

    """ END domain specific functions """
