#Author: Andrea Urgolo
import json
import decimal

""" Utility functions """        

def parse_json(str):
	try:
		from StringIO import StringIO
		io = StringIO(str)
	except ImportError:
			from io import BytesIO
			io = BytesIO(str)

	return json.load(io)

def encode_json(obj):
    return json.dumps(obj, cls=DecimalEncoder)
    
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)
