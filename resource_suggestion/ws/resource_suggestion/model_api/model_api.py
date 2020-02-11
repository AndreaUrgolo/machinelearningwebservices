# -*- coding: utf-8 -*-
import importlib
import pandas as pd 
import numpy as np
import os.path
import random
import time
from scipy.stats.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import linear_model, svm, preprocessing
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn import preprocessing 
from flask import Flask, flash, render_template, request, session, abort, jsonify, make_response, redirect, url_for, current_app
from datetime import timedelta
from datetime import datetime
from functools import update_wrapper
import json
from sklearn.externals import joblib

import platform
import multiprocessing

# from our libs
from config.app_config import *
from libs.utils.json_utils import *
from libs.io.db.mongodb_adapter import *

# X TEST ONLY (non usare in produzione)
#np.random.seed(1) # toglie randomness

## MAIN ENTRY POINT EXECUTION 
app = Flask(__name__)

# apply settings from config/app_config.py
app.config.update(APP_CONFIG)
app.config.update(DB_CONFIG)

db_adapter=MongoDBAdapter.create_data_adapter(config=DB_CONFIG)#, batch_size=BATCH_SIZE)

res_profiles=dict()

def init_data():
    """ Import and prepare application models data """
    global dataset, db_adapter
    
    print("Reading the dataset")

    # use this for test env
    dataset = pd.read_csv(DATASET_FILE, sep=CSV_SEPARATOR, header=0, encoding=CSV_ENCODING, engine='python')
    
    set_columns_type(dataset)
    
    print('DONE')
    
    #dataset.info()


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
	
	
### WEBSERVICE ROUTES 

@app.route("/")
def index():
    return make_response('Welcome, the system is up and running!', 200)

@app.route("/dataset")
def dataset():
    return dataset.to_json(orient='records', force_ascii=False)

@app.route("/operative-centers")
def operative_centers():
    #return jsonify(data_manager.get_oc_ids())
    return jsonify(get_ocs())

@app.route("/predict", methods=['POST', 'OPTIONS'])
@crossdomain(origin='*')
def predict():
    global dataset, res_profiles
    data = request.json

    oc = int(data['oc'])
    rs = data['rs']
    odl = data['odl']
    res = int(data['res'])

    cod = dataset[dataset.IDCENTROOPERATIVO == oc]
    train, test = train_test_split(cod, test_size=0.30, random_state=12345678)

    row = test[(test.CODICEODL==odl) & (test.IDRISORSA==res)].head(1)
    
    # load feature used for classification
    features = open(MODELS_PATH + ML_TASK + '_' + str(oc)+ '_features.txt', 'r', encoding=CSV_ENCODING).read().rstrip('\n').split(CSV_SEPARATOR)
    dummies = open(MODELS_PATH + ML_TASK + '_' + str(oc)+ '_dummies.txt', 'r', encoding=CSV_ENCODING).read().rstrip('\n').split(CSV_SEPARATOR)    

    #prepare X data for classification
    X=row[features]

    X_dummies = pd.get_dummies(X)

    X_clean = pd.DataFrame(columns=dummies)
    X_clean = X_clean.append(X_dummies).fillna(0)[dummies]

    set_columns_type(X_clean)

    # load model for classification
    filename = MODELS_PATH + ML_TASK + '_'+ str(oc) +'.joblib'
    model = joblib.load(filename)    

    pred = model.predict(X_clean)[0]

    #resources_prof = get_res_profiles(rs, dataset)
    resources_prof = res_profiles[str(oc)]
    resources_prof = resources_prof.reset_index(drop=True)

    if len(resources_prof[resources_prof.IDRISORSA==int(pred)].index) <= 0:
        print('Risorsa predetta non presente: %d' % pred)
        random.shuffle(rs)
        results = pd.DataFrame(zip(rs[0:10], [0.01]*10), columns=['id', 'score'])
        response = {
            'results' : results.to_dict(orient='records')
        }
        return jsonify(response)

    similarities, indices = findksimilaritems2(resources_prof[resources_prof.IDRISORSA==int(pred)].index[0],resources_prof,'cosine',9)

    index = indices[0]
    suggested_rs = resources_prof.iloc[index]['IDRISORSA'].tolist()

    suggested_rs = suggested_rs
    similarities = similarities.tolist()

    results = pd.DataFrame(zip(suggested_rs, similarities), columns=['id', 'score'])

    #results = pd.DataFrame(zip(rs[0:10], [1.0]*10), columns=['id', 'score'])

    response = {
         'results' : results.to_dict(orient='records')
    }
    return jsonify(response)


### END WEBSERVICE ROUTES 


if __name__ == "__main__":
    if platform.system() != 'Windows':
        multiprocessing.set_start_method('forkserver')
    init_data()
    app.run(host='0.0.0.0', debug=True, port=5057, threaded=True, use_reloader=False)
