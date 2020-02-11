#Author: Andrea Urgolo @ Space1 
import glob
import sys
from os import path
import pandas
from datetime import datetime

#import from parent dir
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))    
from data_adapter import DataAdapter

class CSVAdapter(DataAdapter):
    """
        Models a CSV file data adapter
        
    """

    #Class Constants
    EXTENSION = 'csv'
    
    # defaults
    DEFAULT_ENCODING = 'utf-8'
    DEFAULT_SEPARATOR = '|'
    
    def __init__(self, batch_size=None, folder, separator=DEFAULT_SEPARATOR, encoding=DEFAULT_ENCODING):
        self.batch_dir = folder
        self.separator= separator
        self.encoding = encoding
        super().__init__(batch_size)
        
    def get_last_batch(self):
        """ Gets the las batch (filename, records) """
        list_of_files = glob.glob(self.batch_dir+ '*.' + type(self).EXTENSION) 
        if len(list_of_files)<0: 
            return (None, None)
        latest_file = max(list_of_files)
        return (latest_file, pandas.read_csv(latest_file, sep=self.separator, encoding=self.encoding))
    
    def save_batch(self, batch_file, batch_data):
        """ Save the batch in the given CSV batch file """
        batch_data.to_csv(batch_file, sep=self.separator, encoding=self.encoding, index=False)
        
    def save_new_record(self, record):
        batch_file, batch = self.get_last_batch()
        if batch is None or len(batch.index) > self.batch_size :
            batch_file = self.get_new_filename()
            batch=record
        self.save_batch(batch_file, batch.append(record, ignore_index=False))
         
    def get_new_filename(self):
        return self.batch_dir + datetime.now()+'.'+ type(self).EXTENSION
        
