# app config
APP_CONFIG=dict(
    SECRET_KEY="SECRET_KEY",
    WTF_CSRF_SECRET_KEY="WTF_CSRF_SECRET_KEY")

# Webservice config
WS_URL="http://localhost:5054" # modulo del modello
HOUSES_REQ = "/houses"
PREDICT_REQ = "/predict/"
SIMILAR_REQ = "/similar/"
SEND_REAL_Y_REQ = "/store-real-y/"

# Application domain config
MILES_KM_RATE = 0.621371
DOLLAR93_EURO_RATE = 1.37 * 5 
PRICE_SIM_TOLLERANCE = 40000
