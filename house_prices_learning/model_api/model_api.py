import pandas
import numpy as np
import os.path
import time
from scipy.stats.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import linear_model, svm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from flask import Flask, flash, render_template, request, session, abort, jsonify, make_response, redirect, url_for
import pickle # save models
import json
from StringIO import StringIO

# from our libs
from config.app_config import *
from libs.utils.json_utils import *
from libs.io.db.mongodb_adapter import *

app = Flask(__name__)

app.config.update(APP_CONFIG)
app.config.update(DB_CONFIG)

# X TEST ONLY (non usare in produzione)
#np.random.seed(1) # toglie randomness

# Importa i dati
def init_data():
    global dataset, dataset_train, dataset_test, db_adapter
    #dataset = pandas.read_csv(DATASET_FILE, sep="|", encoding=CSV_ENCODING)
    dataset = db_adapter.load_dataframe(DATASET_COLLECTION, DATASET_COLUMNS)
    
    ### init_model 
        #if os.path.isfile(DATASET_TRAIN_FILE) and os.path.isfile(DATASET_TEST_FILE) and os.path.isfile(DATASET_TRAINSET_FILE) and os.path.isfile(DATASET_TRAIN_LABELS_FILE) and os.path.isfile(DATASET_TESTSET_FILE) and os.path.isfile(DATASET_TEST_LABELS_FILE) and os.path.isfile(TRAINED_MODEL_FILE) :
    if db_adapter.is_present(DATASET_TRAINSET_COLLECTION):
        print("Caricamento dati!!!!")
        load_all()
    else:
        prepare_datasets()
        train_model(trainset, train_labels)
        print("Salvataggio dati!!!!")
        save_all()

#TODO rivedere
#def generate_report():
    #global trainset, train_labels, testset, test_labels
    
    #load_model() # load clean data
    
    #report = pandas.DataFrame(columns=['model_description', 'accuracy', 'mae', 'mse', 'rmse'])
    
    ## test linear Ridge regression
    #alphas =[10, 1, 0.1, 0.01, 0.001, 0.0001, 0.000001] # reg
    #for alpha in alphas:
        #train_model_linear(trainset, train_labels, alpha)
        #ev = evaluate_model(testset, test_labels)
        #report=report.append(generate_report_row(ev), ignore_index=False)
    
    ## test linear SVR
    #train_model_SVR(trainset, train_labels)
    #ev = evaluate_model(testset, test_labels)
    #report=report.append(generate_report_row(ev), ignore_index=False)

    ### test linear Ridge regression + RFECV
    ##train_model_linear_rfe(trainset, train_labels)
    ##ev = evaluate_model(testset, test_labels)
    ##report=report.append(generate_report_row(ev), ignore_index=False)
    
    ## test linear Ridge regression + PCA
    #for n_feat in range(len(trainset.columns)-1):
        #load_model() #clean
        #train_model_linear_pca(trainset, train_labels, n_feat+1)
        #ev = evaluate_model(testset, test_labels)
        #report=report.append(generate_report_row(ev), ignore_index=False)
    
    ## test ExtraTreesRegression (Best!)
    #load_model() #clean 
    
    #trainset = trainset.drop(['NOX', 'dummy_CHAS', 'RAD', 'ZN'], axis=1)
    #testset = testset.drop(['NOX', 'dummy_CHAS', 'RAD', 'ZN'], axis=1)
    
    
    #for n_feat in range(len(trainset.columns)):
        #load_model() #clean 
    
        #trainset = trainset.drop(['NOX', 'dummy_CHAS', 'RAD', 'ZN'], axis=1)
        #testset = testset.drop(['NOX', 'dummy_CHAS', 'RAD', 'ZN'], axis=1)
    
        #train_model_ETR(trainset, train_labels, n_feat+1)
        #ev = evaluate_model(testset, test_labels)
        #report=report.append(generate_report_row(ev), ignore_index=False)

        
    #load_model()
    #trainset = trainset.drop(['NOX', 'dummy_CHAS', 'RAD', 'ZN'], axis=1)
    #testset = testset.drop(['NOX', 'dummy_CHAS', 'RAD', 'ZN'], axis=1)
    ###best
    #train_model_ETR(trainset, train_labels, 7)
    #ev = evaluate_model(testset, test_labels)
    #report=report.append(generate_report_row(ev), ignore_index=False)
    
    ##clean
    #load_model()
    
    #return report

#TODO rivedere
#def save_best_model_data():
    #global trainset, testset, dataset_test
    ##best model Extra Trees Regressor
    #load_model() #get clean data
    #trainset = trainset.drop(['NOX', 'dummy_CHAS', 'RAD', 'ZN'], axis=1)
    #testset = testset.drop(['NOX', 'dummy_CHAS', 'RAD', 'ZN'], axis=1)
    #n_feat=7
    #train_model_ETR(trainset, train_labels, n_feat) 
    
    #test_y = test_labels.tolist()
    #pred_y = model.predict(testset);

    ## e' gia' nell'ultima riga del report
    ##accuracy = model.score(testset, test_y) 
    ##mae = mean_absolute_error(test_y, pred_y)
    ##mse = mean_squared_error(test_y, pred_y)
    ##rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    
    #trainset.to_csv(BEST_MODEL_TRAINSET_FILE, sep="|", encoding=CSV_ENCODING, index=False)
    ##save model
    #with open(BEST_MODEL_FILE, 'wb') as file:
        #pickle.dump(model, file)
        
    #best_testset = pandas.concat((testset, pandas.Series(test_y), pandas.Series(pred_y)),  axis=1)
    #best_testset.columns.values[len(best_testset.columns)-2]  = 'price'
    #best_testset.columns.values[len(best_testset.columns)-1]  = 'prediction'
    
    #best_testset.to_csv(BEST_MODEL_TESTSET_FILE, sep="|", encoding=CSV_ENCODING, index=False)
    
    ## clean
    #load_model()

#TODO rivedere
#def generate_report_row(ev):
    #row = pandas.DataFrame(index=np.arange(1), columns=['model_description', 'accuracy', 'mae', 'mse', 'rmse'])
    #row['model_description']=str(model)
    #row['accuracy']=ev['accuracy']
    #row['mae']=ev['mae']
    #row['mse']=ev['mse']
    #row['rmse']=ev['rmse']
    #app.logger.info(row)

    #return row

#TODO rivedere
#def save_report(report):    
    ##export to csv
    #report.to_csv(REPORT_FILE, sep="|", encoding=CSV_ENCODING, index=False)

def get_similar(data, n=10):    
    """" Given an item data, returns:
            - predicted value, 
            - similar n items in the testset
            - similarity values related to the n similar items
    """        
    dataset = trainset.append(testset, ignore_index=False)
    data = fix_missing_values(data, dataset)

    data_labels = train_labels.tolist() +  test_labels.tolist()    

    #prediction
    x = np.reshape(pandas.Series(data, index=trainset.columns).values, (1,-1))
    pred = model.predict(x).tolist()[0]

    #get similar items in dataset
    cor=[0.0]*len(dataset.index)
    for i in range(len(cor)):
        cor[i] = pearsonr(dataset.iloc[i,0:], pandas.Series(data, index=dataset.columns).values)[0]
        
    res_idx = np.argsort(cor)[::-1][0:n]
    
    # returns: (predicted value, similar items in the trainset, similarity values related to similar items) 
    return ({
        'prediction': pred, 
        'sim_items': np.asmatrix(dataset)[res_idx, 0:].tolist(), 
        'sim_items_columns': dataset.columns.tolist(),
        'sim_values': np.asarray(cor)[res_idx].tolist(),
        'sim_items_y': np.asarray(data_labels)[res_idx].tolist(), 
        'fixed_data': data
    })


def fix_missing_values(data, dataset, n=10):
    
    #remove unexpected cols from userdata
    tmp=dict(data)
    for c in data:
        if not c in dataset.columns:
            del tmp[c]
    data=tmp
    
    #find missing cols
    missing_cols = list()
    for c in dataset.columns:
        if not c in data:
            missing_cols.append(c)

    #remove missing columns from dataset
    clean_dataset = dataset.drop(missing_cols,axis=1)   
    
    #remove missing elements from userdata
    clean_data=dict(data)
    for c in data:
        if not c in clean_dataset.columns:
            del clean_data[c]
    
    #get similar items in dataset 
    cor=[0.0]*len(clean_dataset.index)
    for i in range(len(cor)):
        cor[i] = pearsonr(clean_dataset.iloc[i,0:], pandas.Series(clean_data, index=clean_dataset.columns).values)[0]
        
    res_idx = np.argsort(cor)[::-1][0:n]
    #similar_items = np.asmatrix(dataset)[res_idx, 0:].tolist()
    
    for c in missing_cols :
        val_sum = 0
        c_i =  dataset.columns.get_loc(c)
        for i in res_idx :
            val_sum += dataset.iloc[i,c_i]
        data[c] = val_sum / n
    
    return data 


""" DATASET HANDLING """

def prepare_datasets():
    global dataset_train, dataset_test, trainset, train_labels, testset, test_labels
    dataset_train = dataset.sample(frac=TRAIN_FRACTION)
    dataset_test = dataset.loc[dataset.index.difference(dataset_train.index)]
    trainset, train_labels = prepare_dataset(dataset_train)
    testset, test_labels = prepare_dataset(dataset_test)

def prepare_dataset(dataset):
    x = dataset[X_COLUMNS]
    y = dataset[Y_COLUMN]
    return (x, y)


def regenerate_datasets():
    global db_adapter
    db_adapter.clear_data(DATASET_TRAINSET_COLLECTION)
    init_data()    

def restore_dataset():
    """ Recover the original dataset """
    global db_adapter
    
    db_adapter.clear_dataframe_with_field(DATASET_COLLECTION, 'timestamp')
    regenerate_datasets()
    
""" MODEL TRAINING """

# Call the training algorithm
def train_model(xTrain, yTrain): 
    global model    
    model = model.fit(xTrain, yTrain)

# Learning algorithm
def retrain_model(k=DEFAULT_KFOLD):
    global db_adapter, trainset, train_labels
    
    old_acc = db_adapter.get_best_model_info(TRAINED_MODEL_COLLECTION)['accuracy']
    
    if k > 1:# k-folds cross-validation
        eval=cross_validation(k)
    else: # k <= 1 no cross validation 
        # separa batch in batch_train e batch_test
        prepare_datasets()
        train_model(trainset, train_labels)
        eval=evaluate_model(testset, test_labels)
    
    if eval['accuracy'] >= old_acc:
        app.logger.info("Accuracy maggiore!!!")
        app.logger.info(eval['accuracy'])
        save_model(eval)
    else :
        app.logger.info("Accuracy peggiore!!!")
        app.logger.info(eval['accuracy'])
        load_model()


def cross_validation(k):
    global dataset, model
    
    evals = [None]*k
    kf = KFold(n_splits=k, shuffle = True)
    
    best_model = None
    best_acc = -100.0
    
    i=0
    for train_index, test_index in kf.split(dataset):
        xTrain = dataset.loc[dataset.index[train_index], X_COLUMNS]
        yTrain = dataset.loc[dataset.index[train_index], Y_COLUMN]
        
        xTest = dataset.loc[dataset.index[test_index], X_COLUMNS]
        yTest = dataset.loc[dataset.index[test_index], Y_COLUMN]
        
        train_model(xTrain, yTrain)

        evals[i]=evaluate_model(xTest, yTest)
        
        if evals[i]['accuracy'] > best_acc:
            best_acc = evals[i]['accuracy']
            best_model = model
        i+=1
    
    model = best_model
    #return {"accuracy":0.0} # dummy NON salva il modello
    return prepare_k_cross_eval_data(evals, model)
    
def prepare_k_cross_eval_data(evals, model):
    """ prepare eval dict """
    eval=evals[0]
    accuracy = 0
    mae = 0
    mse = 0
    rmse = 0
    
    for ev in evals:
        accuracy += ev['accuracy']
        mae += ev['mae']
        mse += ev['mse']
        rmse += ev['rmse']
        
    eval['accuracy'] = accuracy / len(evals)
    eval['mae'] = mae / len(evals)
    eval['mse'] = mse / len(evals)
    eval['rmse'] = rmse / len(evals) 
    return eval


def add_new_record(data):
    global db_adapter, dataset
    data['timestamp'] = time.time()
    db_adapter.save_new_record(data)
    
    #x_data = { k: record[k] for k in x_columns }
        
    batch = db_adapter.get_last_batch()
    #app.logger.info("Last batch")
    #app.logger.info(batch)
        
    if(db_adapter.is_batch_ready()):
        dataset=db_adapter.load_dataframe(DATASET_COLLECTION)
        retrain_model()

    
# Contiene l'algoritmo di Training per Extra Tree Regressor
def train_model_ETR(xTrain, yTrain, max_features=6):
    global model
    model = ExtraTreesRegressor(max_features=max_features, criterion="mae") #default criterion="mse"
    model.fit(xTrain, yTrain)
    
# Contiene l'algoritmo di Training Linear Regression
def train_model_linear(xTrain, yTrain, alpha=0.0001):
    global model
    #model = linear_model.LinearRegression()
    model = Ridge(alpha=alpha)
    model.fit(xTrain, yTrain)

# Contiene l'algoritmo di Training Linear Regression
def train_model_linear_rfe(xTrain, yTrain):
    global model, trainset, testset
    #model = linear_model.LinearRegression()
    model = Ridge(alpha=.0001)
    rfe = RFECV(model, step=1, cv=3)
    rfe.fit(xTrain, yTrain)
    trainset = rfe.transform(trainset)
    testset = rfe.transform(testset)
    model.fit(trainset, yTrain)
    
# Contiene l'algoritmo di Training Linear Regression
def train_model_linear_pca(xTrain, yTrain, n_components=8):
    global model, trainset, testset
    #model = linear_model.LinearRegression()
    model = Ridge(alpha=.0001)
    pca = PCA(n_components=n_components)
    pca.fit(xTrain, yTrain)
    trainset = pca.transform(trainset)
    testset = pca.transform(testset)
    #app.logger.info(np.cumsum(pca.explained_variance_ratio_))
    model.fit(trainset, yTrain)


# Contiene l'algoritmo di Training Multi Level Perceptron
def train_model_MLP(xTrain, yTrain):
    global model
    #model = MLPRegressor(activation='relu', hidden_layer_sizes=120, alpha=0.001, batch_size=2)
    #model = MLPRegressor(activation='logistic', 
                         #hidden_layer_sizes=1299, 
                         #alpha=0.001, 
                         #batch_size=2)
    model = MLPRegressor(
        activation='logistic', 
        hidden_layer_sizes=1299, 
        alpha=0.001, 
        batch_size=2,
        solver='adam',
        #random_state=9
        #learning_rate='adaptive',
        #max_iter=100
        #learning_rate_init=0.01
        )
    model.fit(xTrain, yTrain)
    
# Contiene l'algoritmo di Training SVR
def train_model_SVR(xTrain, yTrain):
    global model
#    model = SVR();
#    model = SVR(kernel='sigmoid',degree=2,gamma=0.5,probability=False,shrinking=True)
#    model = SVR(kernel='rbf',degree=1,gamma=0.5,probability=False,shrinking=True)
#    model = SVR(kernel='poly', degree=0.5,gamma=0.8,probability=False,shrinking=True)
#    model = SVR(kernel='poly', degree=6,gamma=0.8,probability=False,shrinking=True)
    #model = SVR(kernel='poly', degree=2,gamma=0.2,probability=False,shrinking=True)
    #model = SVR(kernel='rbf', degree=3, C=1.0, shrinking=True)
    model = SVR(kernel='linear', C=5.0, epsilon=0.1, tol=0.001,shrinking=True)
    model.fit(xTrain,yTrain)

def save_model(eval_kw=None):
    """ Saves current model """
    global db_adapter, testset, test_labels, model
    if eval_kw is None:
        eval_kw = evaluate_model(testset, test_labels)
    db_adapter.save_model(TRAINED_MODEL_COLLECTION, eval_kw, model, overwrite=False)

def save_all():
    """ Saves current model + all datasets """
    global db_adapter, dataset, dataset_train, dataset_test, trainset, train_labels, testset, test_labels, model
    
    #db_adapter.save_dataframe(DATASET_COLLECTION, dataset) #orig dataset
    db_adapter.save_dataframe(DATASET_TRAIN_COLLECTION, dataset_train)
    db_adapter.save_dataframe(DATASET_TEST_COLLECTION, dataset_test)
    db_adapter.save_dataframe(DATASET_TRAINSET_COLLECTION, pandas.DataFrame(trainset))
    db_adapter.save_dataframe(DATASET_TRAIN_LABELS_COLLECTION, pandas.DataFrame(train_labels))    
    db_adapter.save_dataframe(DATASET_TESTSET_COLLECTION, pandas.DataFrame(testset))
    db_adapter.save_dataframe(DATASET_TEST_LABELS_COLLECTION, pandas.DataFrame(test_labels))
    db_adapter.save_model(TRAINED_MODEL_COLLECTION, evaluate_model(testset, test_labels), model, overwrite=False)
    
    
def load_model():
    global db_adapter, model
    model = db_adapter.load_best_model(TRAINED_MODEL_COLLECTION)


def load_all():
    global db_adapter, dataset, dataset_train, dataset_test, trainset, train_labels, testset, test_labels, model
    
    #dataset = db_adapter.load_dataframe(DATASET_COLLECTION, DATASET_COLUMNS)
    dataset_train = db_adapter.load_dataframe(DATASET_TRAIN_COLLECTION, DATASET_COLUMNS)
    dataset_test = db_adapter.load_dataframe(DATASET_TEST_COLLECTION, DATASET_COLUMNS)
    trainset = db_adapter.load_dataframe(DATASET_TRAINSET_COLLECTION, X_COLUMNS)
    train_labels = db_adapter.load_dataframe(DATASET_TRAIN_LABELS_COLLECTION).iloc[0:,0] 
    testset = db_adapter.load_dataframe(DATASET_TESTSET_COLLECTION, X_COLUMNS)
    test_labels = db_adapter.load_dataframe(DATASET_TEST_LABELS_COLLECTION).iloc[0:,0]
    model = db_adapter.load_best_model(TRAINED_MODEL_COLLECTION)
    
def evaluate_model(test_x, test_y, get_prediction=False):
    global model 
    test_y = test_y.tolist()
    pred_y = model.predict(test_x)
    #pred_y = [int(round(x)) for x in pred_y]
    #accuracy = accuracy_score(test_y, pred_y) 
    accuracy = model.score(test_x, test_y) 
    mae = mean_absolute_error(test_y, pred_y)
    mse = mean_squared_error(test_y, pred_y)
    rmse = np.sqrt(mean_squared_error(test_y, pred_y))
    if get_prediction:
        diff = [(a_i, b_i) for a_i, b_i in zip(pred_y, test_y)] 
        return { 'accuracy': accuracy, 'mse': mse, 'rmse': rmse, 'mae': mae,\
                'y_pred_test': diff, 'model_description': str(model), 'timestamp':time.time() }
    else:
        return { 'accuracy': accuracy, 'mse': mse, 'rmse': rmse, 'mae': mae,\
                 'model_description': str(model), 'timestamp':time.time() }
        

# contiene l'algoritmo di predizione
def get_prediction(x):    
    return model.predict(x).tolist()[0]


@app.route("/")
def index():
    #return redirect(url_for('movies')) 
    link_list='<p>Available services:</p><ul>'+ \
            '<li><a href="'+url_for('dataset')+'">Houses list</a></li>'+ \
            '<li>Model training<ul>'+ \
                    '<li><a href="'+url_for('do_train_model_etr')+'">Extra Tree Regressor</a> (<a href="'+url_for('model_feature_importances') +'">explain</a>)</li>'+ \
                    '<li><a href="'+url_for('do_train_model_linear')+'">Linear Regression (<a href="'+url_for('model_weights') +'">weights</a>)</a></li>'+ \
                    '<li><a href="'+url_for('do_train_model_linear_pca')+'">Linear Regression + PCA (<a href="'+url_for('model_weights') +'">weights</a>)</a></li>'+ \
                    '<li><a href="'+url_for('do_train_model_linear_rfe')+'">Linear Regression + RFE (<a href="'+url_for('model_weights') +'">weights</a>)</a></li>'+ \
                    '<li><a href="'+url_for('do_train_model_mlp')+'">Multi Level Perceptron Neural Network</a></li>'+ \
                    '<li><a href="'+url_for('do_train_model_svr')+'">SVR (<a href="'+url_for('svr_weights') +'">weights</a>)</a></li>'+ \
                '</ul>'+ \
            '<li><a href="'+url_for('do_save_model')+'">Save Model</a></li>'+ \
            '<li><a href="'+url_for('do_load_model')+'">Load Model</a></li>'+ \
            '<li><a href="'+url_for('do_model_eval')+'">Evaluate Model</a></li>'+ \
            '<li><a href="'+url_for('do_regenerate_datasets')+'">Regenerate Datasets</a></li>'+ \
            '<li><a href="'+url_for('do_restore_datasets')+'">Restore the Original Dataset</a></li>'+ \
        '</ul>'
    
    #'<li><a href="'+url_for('report')+'">Save report</a></li>'+ 
    #'<li><a href="'+url_for('do_save_best_model')+'">Save best model data</a></li>'+ 

    return make_response("<p>Welcome, the system is working!!!</p>"+link_list, 200)

@app.route("/dataset/")
def dataset():
    return make_response(dataset.to_json(orient='records', force_ascii=False), 200)

@app.route("/traindata/")
def trains():
    return make_response(dataset_train.to_json(orient='records', force_ascii=False), 200)

@app.route("/testdata/")
def tests():
    return make_response(dataset_test.to_json(orient='records', force_ascii=False), 200)

@app.route("/testset/")
def testset():
    return make_response(pandas.DataFrame(testset).to_json(orient='records', force_ascii=False), 200)

@app.route("/test-labels/")
def test_labels():
    return make_response(pandas.Series(test_labels).to_json(orient='records', force_ascii=False), 200)

@app.route("/model-eval/")
def do_model_eval():
    return make_response(jsonify(evaluate_model(testset, test_labels)), 200)

@app.route("/save-model/")
def do_save_model():
    save_model()
    return make_response("Model saved correctly", 201)

#TODO rivedere
#@app.route("/save-best-model/")
#def do_save_best_model():
    #save_best_model_data()
    #return make_response("Best model saved correctly", 200)

@app.route("/load-model/")
def do_load_model():
    load_model()
    return make_response("Model loaded correctly", 200)

@app.route("/regenerate-datasets/")
def do_regenerate_datasets():
    regenerate_datasets()
    return make_response("Datasets regenerated correctly", 200)


@app.route("/restore-datasets/")
def do_restore_datasets():
    restore_dataset()
    return make_response(jsonify({"message":"Datasets restored correctly"}), 200)


@app.route("/predict/", methods=['POST']) 
def predict():
    data = request.json

    # fix missing values in data exploiting similar items 
    data = fix_missing_values(data, trainset.append(testset, ignore_index=False))
    
    item = np.reshape(pandas.Series(data, index=trainset.columns).values, (1,-1))
    
    pred = get_prediction(item)
    res = {'prediction': pred,
           'fixed_data': data}
    return make_response(jsonify(res), 200)

@app.route("/similar/", methods=['POST']) 
def similar():
    data = request.json
    sims = get_similar(data, 10) # last param means N similar items
    return make_response(jsonify(sims), 200)

@app.route("/store-real-y/", methods=['POST']) 
def do_store_real_y():
    data = request.json
    data[Y_COLUMN]=data['real_y']
    data.pop('real_y', None)
    
    add_new_record(data)
    return make_response(jsonify({"message": "OK"}), 201)


#TODO rivedere
#@app.route("/report/")
#def report():
    #report = generate_report()
    #save_report(report)
    #return make_response(report.to_json(orient='records', force_ascii=False), 200)


# WS FOR TRAINING 

@app.route("/etr/train/")
def do_train_model_etr():
    load_all()
    train_model_ETR(trainset, train_labels)
    return make_response(jsonify({"message": "OK"}), 200)

@app.route("/etr/feature-importances/")
def model_feature_importances():
    output=[None]*len(model.feature_importances_)
    for i in range(len(model.feature_importances_)) :
        output[i]={}
        output[i][''+trainset.columns[i]]=model.feature_importances_[i]
        
    return make_response(jsonify(output), 200)

@app.route("/linear/train/")
def do_train_model_linear():
    load_all()
    train_model_linear(trainset, train_labels)
    return make_response(jsonify({"message": "OK"}), 200)

@app.route("/linear/train-rfe/")
def do_train_model_linear_rfe():
    load_all()
    train_model_linear_rfe(trainset, train_labels)
    return make_response(jsonify({"message": "OK"}), 200)

@app.route("/linear/train-pca/")
def do_train_model_linear_pca():
    load_all()
    train_model_linear_pca(trainset, train_labels)
    return make_response(jsonify({"message": "OK"}), 200)

@app.route("/linear/weights/")
def model_weights():
    return make_response(jsonify(model.coef_.tolist()), 200)

@app.route("/mlp/train/")
def do_train_model_mlp():
    load_all()
    train_model_MLP(trainset, train_labels)
    return make_response(jsonify({"message": "OK"}), 200)

@app.route("/svr/train/")
def do_train_model_svr():
    load_all()
    train_model_SVR(trainset, train_labels)
    return make_response(jsonify({"message": "OK"}), 200)

@app.route("/svr/weights/")
def svr_weights():
    return make_response(jsonify(model.coef_.tolist()), 200)


if __name__ == "__main__":
    db_adapter=MongoDBAdapter.create_data_adapter(config=DB_CONFIG, batch_size=DEFAULT_BATCH_SIZE)
    init_data()
    app.run(host='0.0.0.0', debug=True, port=5054)
