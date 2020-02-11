# app config
APP_CONFIG=dict(
    SECRET_KEY="SECRET",
    WTF_CSRF_SECRET_KEY="SECRET",

    # Webservice config
	WS_URL="http://localhost:5057", # ws del modello in locale
	DATASET_REQ = "/dataset",
	OPERATIVE_CENTERS_REQ = "/operative-centers",
	OC_DATE_REQ = "/oc-date",
	OC_DATA_REQ = "/oc-data",
	PREDICT_REQ = "/predict",

	DATA_CHARSET ='ISO-8859-1'
)

# Application domain config
