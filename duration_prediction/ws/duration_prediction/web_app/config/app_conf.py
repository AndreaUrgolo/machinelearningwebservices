# app config
APP_CONFIG=dict(
    SECRET_KEY="SECRET",
    WTF_CSRF_SECRET_KEY="SECRET",

    # Webservice config
	WS_URL="http://localhost:5058", # ws del modello in locale
	DATASET_REQ = "/dataset",
	OPERATIVE_CENTERS_REQ = "/operative-centers",
	OC_DATE_REQ = "/oc-date",
	OC_DATA_REQ = "/oc-data",
	PREDICT_REQ = "/predict",
	OPERATORS_RANK_REQ = "/ops-rank/",
	SEND_REAL_Y_REQ = "/store-real-y/",
	SEND_LOGIN_REQ= "/login",
	MODELS_REQ = "/models",
	EVAL_MODELS_REQ = '/eval-models',
	USE_MODEL_REQ='/use-models',
	ALGORITHMS_REQ='/algorithms',
	SETTINGS_REQ = '/settings',

	DATA_CHARSET ='ISO-8859-1'
)

# Application domain config
