#Author: Andrea Urgolo @ Space1 
class DataAdapter(object): #Abstract
    
    DEFAULT_BATCH_SIZE = 100
    
    def __init__(self, batch_size=DEFAULT_BATCH_SIZE):
        self.batch_size=batch_size

    def is_present(self, source):
        raise NotImplementedError()

    def save_new_record(self, record):
        raise NotImplementedError()

    def clear_data(self, source):
        raise NotImplementedError()

    def clear_dataframe_with_field(self, where, field):
        raise NotImplementedError()
    
    def save_dataframe(self, where, data, overwrite=True):
        raise NotImplementedError()

    def load_dataframe(self, source, columns=None):
        raise NotImplementedError()
    
    def save_model(self, where, model_kw, data, overwrite):
        raise NotImplementedError()
    
    def load_last_model(self, source):
        raise NotImplementedError()
    
    def load_best_model(self, source):
        raise NotImplementedError()
    
    def get_best_model_info(self, source):
        raise NotImplementedError()
    
    def get_last_batch(self):
        raise NotImplementedError()
    
    def is_batch_ready(self):
        raise NotImplementedError()
    
    # factory method
    @staticmethod
    def create_data_adapter(**kwargs):
        raise NotImplementedError()
