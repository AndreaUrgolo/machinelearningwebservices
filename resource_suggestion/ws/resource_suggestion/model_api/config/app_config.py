APP_CONFIG=dict(
    SECRET_KEY="SECRET",
    JSONIFY_PRETTYPRINT_REGULAR=False
)

DB_CONFIG={
    'db': 'res_db',
    'host': 'mongodb',
    'port': 27017,
    'username': 'user',
    'password': 'pwd',
    'authentication_source': 'admin',
    'dataset_source': 'dataset',
    'users_source': 'users',
    'auth_tokens_source': 'auth_tokens',
    'models_source' : 'models',
    'dataset_train_source' : 'dataset_train',
    'dataset_test_source' : 'dataset_test',
    'dataset_trainset_source': 'trainset',
    'dataset_train_labels_source': 'train_labels',
    'dataset_testset_source': 'testset',
    'dataset_test_labels_source': 'test_labels',
    'settings_source': 'settings'
}

AUTH_CONFIG={
    # Key-Value dictionary with a list of allowed routes for each role:
    # - The key correspond to the required role 
    # - The value correspond to the list of allowed routes
    'admin':['/admin-test']
}

# Webservice config
CSV_ENCODING = 'ISO-8859-1'
CSV_SEPARATOR = '§'

# Webservice file config

CUSTOMER_NAME = 'customer'
DATA_PATH = 'data/'
DATASET_FILE = DATA_PATH + CUSTOMER_NAME + '/final/dataset.csv'
MODELS_PATH = DATA_PATH + CUSTOMER_NAME + '/models/'
RESOURCE_COLUMNS_FILE = DATA_PATH + CUSTOMER_NAME + "/final/resource_columns_1.txt"
ML_TASK = 'resource_suggestion'

# Data specific config
Y_COLUMN = 'TARGET_COLUMN'

# Application domain config

# DEFAULT_TOP_N_REC = 10
TRAIN_FRACTION = 0.8
# learner configuration
# BATCH_SIZE  = 100
# KFOLD = 5 # 1 = no cross validaton => only (trainset + testset)
# LEARNING_RATE = 0.4

# ALGORITHMS = [
#     {'name': 'MLPRegressor', 'desc': 'Multi-layer Perceptron Regressor', 'module':'sklearn.neural_network', 'class': 'MLPRegressor'},
#     {'name': 'LinearRegression', 'desc': 'Least squares Linear Regression', 'module':'sklearn.linear_model', 'class': 'LinearRegression'},
#     {'name': 'DecisionTreeRegressor', 'desc':'Decision Tree Regressor', 'module':'sklearn.tree', 'class':'DecisionTreeRegressor'},
#     {'name': 'SVR', 'desc':'Epsilon-Support Vector Regression', 'module':'sklearn.svm', 'class': 'SVR'}
# ]
