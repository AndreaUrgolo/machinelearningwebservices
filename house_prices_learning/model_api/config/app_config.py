APP_CONFIG=dict(
    JSONIFY_PRETTYPRINT_REGULAR=False
)


DB_CONFIG={
    'db': 'houses_db',
    'host': 'localhost',
    'port': 27017,
    'username': 'user',
    'password': 'pwd',
    'authentication_source': 'admin'
}

# config
CSV_ENCODING= "utf-8" #TODO spostare 

# Webservice config
DATASET_FILE = "data/dataset.data.txt"
#DATASET_FILE = "data/dataset.csv"
DATASET_TRAIN_FILE = "data/trained_model/dataset_train.csv"
DATASET_TEST_FILE = "data/trained_model/dataset_test.csv"
DATASET_TRAINSET_FILE = "data/trained_model/trainset.csv"
DATASET_TRAIN_LABELS_FILE = "data/trained_model/train_labels.csv"
DATASET_TESTSET_FILE = "data/trained_model/testset.csv"
DATASET_TEST_LABELS_FILE = "data/trained_model/test_labels.csv"
TRAINED_MODEL_FILE = "data/trained_model/trained_model.pkl"


DATASET_COLLECTION = 'dataset'
DATASET_TRAIN_COLLECTION = 'dataset_train'
DATASET_TEST_COLLECTION = 'dataset_test'
DATASET_TRAINSET_COLLECTION = 'trainset'
DATASET_TRAIN_LABELS_COLLECTION = 'train_labels'
DATASET_TESTSET_COLLECTION = 'testset'
DATASET_TEST_LABELS_COLLECTION = 'test_labels'
TRAINED_MODEL_COLLECTION = 'model'


REPORT_FILE = "data/trained_model/report.csv"
BEST_MODEL_FILE = "data/trained_model/best_model.pkl"
BEST_MODEL_TRAINSET_FILE = "data/trained_model/best_model_trainset.csv"
BEST_MODEL_TESTSET_FILE = "data/trained_model/best_model_testset.csv"

#  Application domain config
DEFAULT_BATCH_SIZE  = 10
DEFAULT_TOP_N_REC = 10
TRAIN_FRACTION = 0.8
DEFAULT_KFOLD = 5 # 1 = no cross validaton => only (trainset + testset)
DATASET_COLUMNS = ['CRIM', 'ZN', 'INDUS', 'dummy_CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Drop columns 
#[] # per il report con tutte le feature
#['NOX', 'dummy_CHAS', 'RAD', 'DIS', 'B', 'ZN']
#['NOX', 'dummy_CHAS', 'RAD', 'AGE', 'DIS']
#['NOX', 'dummy_CHAS', 'RAD', 'ZN'] 
#DROP_COLUMNS = ['ZN','NOX', 'dummy_CHAS', 'RAD', 'B']
X_COLUMNS = ['CRIM', 'INDUS', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'LSTAT']
Y_COLUMN = 'MEDV'
