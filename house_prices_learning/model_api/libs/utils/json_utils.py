#Author: Andrea Urgolo @ Space1 
import json
import decimal
from StringIO import StringIO

""" Utility functions """        

def parse_json(str):
    io = StringIO(str)
    return json.load(io)

def encode_json(obj):
    return json.dumps(obj, cls=DecimalEncoder)
    
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)
