# -*- coding: utf-8 -*-
#Author: Andrea Urgolo
from flask import Flask, flash, render_template, request, session, abort, jsonify, make_response, redirect, url_for, current_app
from datetime import timedelta
from datetime import datetime
from functools import update_wrapper
# import simplejson
# import time
import platform
import multiprocessing

# from our libs
from config.app_config import *
from libs.io.db.mongodb_adapter import *
from libs.io.data_manager import *
from libs.ml.ml_manager import *

## MAIN ENTRY POINT EXECUTION 
app = Flask(__name__)

# apply settings from config/app_config.py
app.config.update(APP_CONFIG)
app.config.update(DB_CONFIG)

# Initialize DB and data manager
db_adapter = MongoDBAdapter.create_data_adapter(config=DB_CONFIG, batch_size=APP_CONFIG['BATCH_SIZE'])
data_manager = DataManager(config=APP_CONFIG, db_adapter=db_adapter)
ml_manager = MLManager(APP_CONFIG, db_adapter, data_manager)

### crossdomain fix
def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, list):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, list):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator
### END crossdomain fix

## WEBSERVICE ROUTES ##

@app.route("/")
def index():
    return make_response('Welcome, the system is up and running!', 200)

@app.route("/dataset")
def dataset():
    return data_manager.get_dataset().to_json(orient='records', force_ascii=False)

@app.route("/operative-centers")
def operative_centers():
    #return jsonify(data_manager.get_oc_ids())
    return jsonify(data_manager.get_ocs())
    
@app.route("/save-all-models")
def save_all_models():
    ml_manager.save_all_models()
    return make_response('Models saved correctly', 201)

@app.route("/delete-all-models")
def delete_all_models():
    db_adapter.clear_data(DB_CONFIG['models_source'])
    return make_response('Models deleted correctly', 201)


@app.route("/oc-model-data", methods=['POST', 'OPTIONS'])
@crossdomain(origin='*')
def oc_model_data():
    data = request.json
    oc = data['oc']

    model, features, dummies = ml_manager.get_model_data(oc)

    if model is not None:
        return jsonify({'features':features, 'dummies':dummies})
    else :
        return make_response('Model not found', 404)
    
@app.route("/oc-date", methods=['POST', 'OPTIONS'])
@crossdomain(origin='*')
def oc_date():
    data = request.json
    oc = data['oc']

    first_date = data_manager.get_first_test_data(oc)
    return jsonify(datetime.fromtimestamp(first_date).strftime("%d-%m-%Y %H:00"))

@app.route("/oc-data", methods=['POST', 'OPTIONS'])
@crossdomain(origin='*')
def oc_data():
    data = request.json
    oc = data['oc']

    ws, rs = data_manager.get_oc_data(oc, 40)

    response = {
        'ws' : ws.to_dict(orient='records'),
        'rs' : rs.to_dict(orient='records')
    }

    return jsonify(response)
    #return simplejson.dumps(response, allow_nan=False, encoding=APP_CONFIG['DATA_ENCODING'])

@app.route("/predict", methods=['POST', 'OPTIONS'])
@crossdomain(origin='*')
def predict():
    data = request.json
    oc = int(data['oc'])
    rs = data['rs']
    res = int(data['res'])
    app = int(data['app'])
    odl = data['odl']

    results = ml_manager.predict_duration(oc, rs, odl, res, app)

    response = {
        'results' : results.to_dict(orient='records')
    }
    return jsonify(response)

if __name__ == "__main__":
    if platform.system() != 'Windows':
        multiprocessing.set_start_method('forkserver')
    data_manager.init_data()
    app.run(host='0.0.0.0', debug=True, port=5058, threaded=True, use_reloader=False)
