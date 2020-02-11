# -*- coding: utf-8 -*-
#Author: Andrea Urgolo
class DataAdapter(object): #Abstract
    
    DEFAULT_BATCH_SIZE = 100
    
    def __init__(self, batch_size=DEFAULT_BATCH_SIZE):
        self.batch_size=batch_size

    def is_present(self, source):
        raise NotImplementedError()

    def save_new_record(self, record):
        raise NotImplementedError()

    def clear_data(self, where):
        raise NotImplementedError()
    
    def save_data(self, where, data, key):
        raise NotImplementedError()

    def save_dict_data(self, where, data):
        raise NotImplementedError()

    def get_data(self, source, query):
        raise NotImplementedError()

    def get_one_data(self, source, query):
        raise NotImplementedError()
    
    def get_one_data_by_id(self, source, id):
        raise NotImplementedError()
    
    def remove_data_by_id(self, source, id):
        raise NotImplementedError()
    
    def clear_dataframe_with_field(self, where, field):
        raise NotImplementedError()
    
    def save_dataframe(self, where, data, overwrite):
        raise NotImplementedError()

    def get_dataframe(self, source, columns):
        raise NotImplementedError()
    
    def save_model(self, **kwargs):
        raise NotImplementedError()
    
    def update_model(self, id, model_kw, model):
        raise NotImplementedError()    

    def get_model_by_id(self, id):
        raise NotImplementedError()
    
    def get_model_info_by_id(self, id):    
        raise NotImplementedError()
        
    def get_last_model(self):
        raise NotImplementedError()
    
    def get_best_model(self):
        raise NotImplementedError()
    
    def get_best_model_info(self):
        raise NotImplementedError()
    
    def get_last_batch(self):
        raise NotImplementedError()
    
    def is_batch_ready(self):
        raise NotImplementedError()
        
    
    """ Auth collections handling """
    def check_role(self, user_id, role): 
        raise NotImplementedError()
        
    def login(self, username, password):
        raise NotImplementedError() 
    
    """ Settings management """
    def load_settings(self):
        raise NotImplementedError() 
    
    def save_settings(self, data):
        raise NotImplementedError() 
    
    """ factory method """
    @staticmethod
    def create_data_adapter(**kwargs):
        raise NotImplementedError()
