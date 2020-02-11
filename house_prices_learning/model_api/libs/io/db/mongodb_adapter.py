#Author: Andrea Urgolo @ Space1 
import glob
import sys
from os import path
import pandas
import pickle # save models
from pymongo import MongoClient

#from datetime import datetime

#import from parent dir
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))    
from data_adapter import DataAdapter


class MongoDBAdapter(DataAdapter):
    """ Models a MongoDB data adapter """

    #Class Constants
    # e.g. EXTENSION = 'csv'
    
    # defaults
    # e.g. DEFAULT_ENCODING = 'utf-8'
    
    def __init__(self, config=None, **kwargs):
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
        
    def clear_data(self, source):
        coll = self.db[source]
        coll.drop()
        
    def clear_dataframe_with_field(self, where, field):
        coll = self.db[where]
        coll.remove({field:{"$exists":True }})

    def save_dataframe(self, where, data, overwrite=True):
        coll = self.db[where]
        if overwrite: 
            coll.drop() # otherwise the data is appended 
        coll.insert_many(data.to_dict('records'))

    def load_dataframe(self, source, columns=None):
        coll = self.db[source]
        if columns is None:
            return pandas.DataFrame(list(coll.find()))
        else:
            return pandas.DataFrame(list(coll.find()), columns=columns)


    def save_model(self, where, model_kw, model, overwrite=True):
        coll = self.db[where]
        if overwrite:
            coll.drop() # otherwise the data is appended
        model_kw['model'] = pickle.dumps(model)
        coll.insert(model_kw)

    def load_last_model(self, source):
        coll = self.db[source]
        model_str=coll.find_one({},{'model':1}, sort = [('timestamp', -1)])
        return pickle.loads(model_str)
    
    def load_best_model(self, source):
        coll = self.db[source]
        model_str=coll.find_one({},{'model':1}, sort = [('accuracy', -1)])
        return pickle.loads(model_str['model'])
    
    def get_best_model_info(self, source):
        coll = self.db[source]
        return coll.find_one({},{'model':0}, sort = [('accuracy', -1)])

    def get_last_batch(self):
        """ Gets the last batch """
        coll = self.db['model']
        last_timestamp=coll.find_one({},{'timestamp':1}, sort = [('timestamp', -1)])['timestamp']
        
        coll = self.db['dataset']
        return pandas.DataFrame(list(coll.find({'timestamp': {'$gte': last_timestamp}}))) 
      
    def is_batch_ready(self):
        if len(self.get_last_batch().index) >= self.batch_size:
            return True
        return False
        
    def save_new_record(self, record):
        coll = self.db['dataset']
        coll.insert(record)
        
        
    @staticmethod
    def create_data_adapter(**kwargs):
        return MongoDBAdapter(**kwargs)
