# -*- coding: utf-8 -*-
#Author: Andrea Urgolo
import sys
from os import path
import pandas as pd
import pickle # save models
from pymongo import MongoClient
from bson.objectid import ObjectId

#from datetime import datetime

#import from parent dir
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))    
from data_adapter import DataAdapter


class MongoDBAdapter(DataAdapter):
    """ Models a MongoDB data adapter """

    #Class Constants
    # e.g. EXTENSION = 'csv'
    # defaults
    # e.g. DEFAULT_ENCODING = 'ISO-8859-1'

    def __init__(self, config=None, **kwargs):
        self.config = config
        self.conn = MongoClient(config['host']+':'+str(config['port']),
            username=config['username'],
            password=config['password'],
            authSource=config['authentication_source'])
        self.db = self.conn[config['db']]
        super(MongoDBAdapter, self).__init__(**kwargs)

    def is_present(self, source):
        """ Check wether a collection is defined and not empty. """
        if source in self.db.collection_names():
            coll = self.db[source]
            if coll.count() > 0:
                return True
        return False

    def clear_data(self, where):
        """ drop the 'where' collection (if present) """
        if where in self.db.collection_names():
            coll = self.db[where]
            coll.drop()
        
    def save_data(self, where, data, key = None):
        """ Save a data in the 'where' collection  """
        coll = self.db[where]
        if key is None:
            coll.insert(data)
        else:
            coll.insert({key: data})

    def save_dict_data(self, where, data):
        """ Save a data in the 'where' collection  """
        coll = self.db[where]
        # bulk = []
        # for key in data:
        #     bulk += [insert_one({key: data[key]})]    
        coll.insert_many([{"key" : key, "value" : data[key]} for key in data])
    
    def get_one_data(self, source, query={}):
        """ Get one document from the 'source' collection, eventually with an optional query """
        if source in self.db.collection_names():
            coll = self.db[source]
            return coll.find_one(query)
        return False

    def get_one_data_by_id(self, source, id):
        """ Given an id, return the relative document from the 'source' collection  """
        if source in self.db.collection_names():
            coll = self.db[source]
            return coll.find_one({'_id': ObjectId(id)})
        return False
    
    def remove_data_by_id(self, source, id):
        """ Given an id, remove the relative document from the 'source' collection """
        if source in self.db.collection_names():
            coll = self.db[source]
            return coll.remove({'_id': ObjectId(id)})
        return False
    
    def get_data(self, source, query={}):
        """ Get all documents from the 'source' collection, eventually with an optional querry """
        if source in self.db.collection_names():
            coll = self.db[source]
            return coll.find(query)
        return None

    def clear_dataframe_with_field(self, where, field):
        """ Remove all documents from the 'where' collection if they have a given field """
        coll = self.db[where]
        coll.remove({field:{"$exists":True }})

    def save_dataframe(self, where, data, overwrite=True):
        """ Save the dataframe 'data' in the 'where' collection. 
            If 'overwrite' is set to True it replaces the entire contents of the collection 
        """
        coll = self.db[where]
        if overwrite:
            coll.drop() # otherwise the data is appended 
        coll.insert_many(data.to_dict('records'))

    def get_dataframe(self, source, columns=None):
        """ Get a dataframe from the 'source' collection, eventually filtered with the optional columns list """
        coll = self.db[source]
        if columns is None:
            return pd.DataFrame(list(coll.find()))
            #return pd.DataFrame.from_records(coll.find())
        else:
            return pd.DataFrame(list(coll.find()), columns=columns)
            #return pd.DataFrame.from_records(coll.find(), columns=columns)

    def save_model(self, **kwargs):
        """ Save the given model data  """
        #print('Salvataggio modello con dati...')
        # for k,v in kwargs.items():
        #     if k !='model':
        #        print(k, v)
        
        coll = self.db[self.config['models_source']]
        coll.insert(kwargs)
        
    def update_model(self, id, model_kw, model):
        coll = self.db[self.config['models_source']]
        model_kw['model'] = pickle.dumps(model)
        r = coll.update({'_id': ObjectId(id)}, model_kw)
        return r['updatedExisting']

    def get_model_by_id(self, id):
        coll = self.db[self.config['models_source']]
        model_str=coll.find_one({'_id': ObjectId(id)},{'model':1})
        if model_str and 'model' in model_str.keys():
            return pickle.loads(model_str['model'])
        else : return False
    
    def get_model_info_by_id(self, id):
        coll = self.db[self.config['models_source']]
        model_str=coll.find_one({'_id': ObjectId(id)},{'model':0})
        if model_str:
            return model_str
        else : return False

    def get_model_by_oc(self, oc):
        coll = self.db[self.config['models_source']]
        model_str=coll.find_one({'oc': oc}, {'model':1, 'features':1, 'dummies':1}, sort = [('timestamp', -1)])
        if model_str and 'model' in model_str.keys():
            model_str['model'] = pickle.loads(model_str['model'])
            return model_str
        else : return False

    def get_last_model(self):
        coll = self.db[self.config['models_source']]
        model_str=coll.find_one({},{'model':1}, sort = [('timestamp', -1)])
        return pickle.loads(model_str)
    
    def get_best_model(self):
        coll = self.db[self.config['models_source']]
        model_str=coll.find_one({},{'model':1}, sort = [('accuracy', -1)])
        return pickle.loads(model_str['model'])
    
    def get_best_model_info(self):
        coll = self.db[self.config['models_source']]
        return coll.find_one({},{'model':0}, sort = [('accuracy', -1)])

    def get_last_batch(self):
        """ Gets the last batch """
        coll = self.db[self.config['models_source']]
        last_timestamp=coll.find_one({},{'timestamp':1}, sort = [('timestamp', -1)])['timestamp']
        
        coll = self.db[self.config['dataset_source']]
        return pd.DataFrame(list(coll.find({'timestamp': {'$gte': last_timestamp}}))) 
      
    def is_batch_ready(self):
        if len(self.get_last_batch().index) >= self.batch_size:
            return True
        return False
        
    def save_new_record(self, record):
        coll = self.db[self.config['dataset_source']]
        coll.insert(record)
        

    """ Auth collections handling """
    def check_role(self, user_id, role): 
        coll = self.db[self.config['users_source']]
        user = coll.find_one({'_id': user_id})
        if(user is None):
            return False
        else:
            return (user['role'] == role)
    
    def login(self, username, password):
        """ Check if the given credential are related to a registered user of the application.
            Returns the user id or False if username or password are not correct
            :param username 
            :param password 
            :return string|bool
        """
        
        coll = self.db[self.config['users_source']]
        user = coll.find_one({'username': username, 'password' : password})
        if(user is None):
            return False
        else:
            return str(user.get('_id'))
        
    def load_settings(self):
        if self.config['settings_source'] in self.db.collection_names():
            coll = self.db[self.config['settings_source']]
            return coll.find_one({})
        return False

    def save_settings(self, data):
        """Saves the data dictionary to the settings collection """
        self.clear_data(self.config['settings_source'])
        coll = self.db[self.config['settings_source']]
        r = coll.insert_one(data)
        return r.inserted_id

    @staticmethod
    def create_data_adapter(**kwargs):
        """ Factory method, returns an instance of the class """
        return MongoDBAdapter(**kwargs)
