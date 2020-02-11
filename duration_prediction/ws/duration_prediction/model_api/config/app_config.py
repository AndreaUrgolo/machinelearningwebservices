CUSTOMER_NAME = 'customer'

APP_CONFIG=dict(
    SECRET_KEY="SECRET",
    JSONIFY_PRETTYPRINT_REGULAR=False,
    JSON_AS_ASCII = False,
    
    # Data representation config
    DATA_ENCODING = 'ISO-8859-1',
    DATA_SEPARATOR = '§',

    # App domain config
    ML_TASK = 'duration_regression',
    CUSTOMER_NAME = CUSTOMER_NAME, 
    DATA_PATH = '../../data/',
    DATASET_FILE = '../../data/'+CUSTOMER_NAME+'/final/dataset.csv',

    CONFIG_FILES = dict(
        COLUMNS_TYPE_FILE = '../../data/'+CUSTOMER_NAME+'/columns_type.txt',
        VALID_COLUMNS_FILE = '../../data/'+CUSTOMER_NAME+'/final/valid_columns_duration_regression_1.txt',
        RESOURCE_COLUMNS_FILE = '../../data/'+CUSTOMER_NAME+'/final/resource_columns_1.txt'
    ),

    # Data specific config
    # DATASET_COLUMNS = [
        # now is "columns_type.keys"
    # ]

    DURATION_OUTLIERS_LIMIT = 1500, # to filter outliers with TEMPOSOLOLAVORO_COMPUTED > DURATION_OUTLIERS_LIMIT

    Y_COLUMN = 'TARGET_COLUMN',

    # Application domain config
    # DEFAULT_TOP_N_REC = 10
    TRAIN_FRACTION = 0.8,
    # # learner configuration
    BATCH_SIZE  = 100
    # KFOLD = 5, # 1 = no cross validaton => only (trainset + testset)
    # LEARNING_RATE = 0.4,

    # ALGORITHMS = [
        # {'name': 'MLPRegressor', 'desc': 'Multi-layer Perceptron Regressor', 'module':'sklearn.neural_network', 'class': 'MLPRegressor'},
        # {'name': 'LinearRegression', 'desc': 'Least squares Linear Regression', 'module':'sklearn.linear_model', 'class': 'LinearRegression'},
        # {'name': 'DecisionTreeRegressor', 'desc':'Decision Tree Regressor', 'module':'sklearn.tree', 'class':'DecisionTreeRegressor'},
        # {'name': 'SVR', 'desc':'Epsilon-Support Vector Regression', 'module':'sklearn.svm', 'class': 'SVR'}
    # ]

)

DB_CONFIG={
    'db': 'duration_db',
    'host': "mongodb",
    'port': 27017,
    'username': 'user',
    'password': 'pwd',
    'authentication_source': 'admin',
    'dataset_source': 'dataset',
    'users_source': 'users',
    'auth_tokens_source': 'auth_tokens',
    'models_source' : 'models',
}
