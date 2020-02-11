import pandas
from flask import Flask, flash, render_template, request, session, abort, jsonify, make_response, redirect, url_for
from flask_bootstrap import Bootstrap
import requests

# from our libraries
from config import *
from libs.forms.house_form import HouseForm
from libs.forms.real_y_form import RealYForm
from libs.utils.json_utils import *
from libs.utils.flask_utils import *

app = Flask(__name__)
app.config.update(APP_CONFIG)
Bootstrap(app)

# preparo dati generi per chiamata WS
def get_houses():
    return parse_json(requests.get(WS_URL + HOUSES_REQ).content)

# contiene l'algoritmo di raccomandazione
def get_prediction(req):
    r = requests.post(WS_URL + PREDICT_REQ, json=req)
    return parse_json(r.content)


# contiene l'algoritmo di raccomandazione
def get_similar(req):
    r = requests.post(WS_URL + SIMILAR_REQ, json=req)
    return parse_json(r.content)


def prepare_house_data(form):
    req = {}

    for f in form :
        try:
            if f.name == 'dis' and f.data  :
                req[''+f.name.upper()] = float(f.data) * MILES_KM_RATE
            elif f.data:
                req[''+f.name.upper()] = float(f.data)
        except ValueError,e:
            # il campo csfr_token non e' float
            #app.logger.info(f.name)
            #app.logger.info(f.data)
            continue
    return req

def prepare_real_y_data(req, form):
    req['real_y'] = float(form.real_y.data)/(DOLLAR93_EURO_RATE*1000)
    return req

def send_real_y(req):
    res=requests.post(WS_URL + SEND_REAL_Y_REQ, json=req).status_code
    #if res >= 200 and res < 300:
    return True
    #return False
    
@app.route("/")
def index():
    if 'house_form' in session:
        form=HouseForm(data=parse_json(session['house_form']))
    else:
        form=HouseForm()
    return render_template('input.html', form=form)

@app.route("/predict/", methods=['GET','POST'])
def predict():
    if request.method=='POST':
        form=HouseForm(request.form)
        session['house_form'] = encode_json(form.data)
    elif 'house_form' in session :
        form=HouseForm(data=parse_json(session['house_form']))
    else :
        return redirect(url_for('index'))        

    if not form.validate():
        render_template('input.html', form=form)

    req_data=prepare_house_data(form)
    session['x'] = req_data
    
    real_y_form=RealYForm()
    
    if 'real_y_form' in session :
        real_y_form = RealYForm(data=parse_json(session['real_y_form']))
    else:
        real_y_form = RealYForm()
    
    price = float(get_prediction(req_data)['prediction']) * DOLLAR93_EURO_RATE
    house_fixed = get_prediction(req_data)['fixed_data']
    return render_template('result.html', house=req_data, labels=form.labels, price=price, house_fixed=house_fixed, real_y_form=real_y_form)

@app.route("/similar/", methods=['GET','POST'])
def similar():

    if request.method=='POST':
        form=HouseForm(request.form)
        session['house_form'] = encode_json(form.data)
    elif 'house_form' in session :
        form=HouseForm(data=parse_json(session['house_form']))
    else :
        return redirect(url_for('index'))

    if not form.validate():
        return render_template('input.html', form=form)
    
    req_data=prepare_house_data(form)
    
    res = get_similar(req_data)
    
    price=float(res['prediction']) * DOLLAR93_EURO_RATE
    sim_items=res['sim_items']
    sim_items_columns=res['sim_items_columns']
    sim_vals=res['sim_values']

    sim_items_y = [float(i) for i in res['sim_items_y']]
    
    for i in range(len(sim_items_y)):
        sim_items_y[i] *= (DOLLAR93_EURO_RATE*1000)
        
    house_fixed=res['fixed_data'] # valori NA sostituiti da valori calcolati con elementi simili
    session['x'] = house_fixed

    if 'real_y_form' in session :
        real_y_form = RealYForm(data=parse_json(session['real_y_form']))
    else :
        real_y_form = RealYForm()
    
    return render_template('result.html', house=req_data, labels=form.labels, price=price, similar_items=sim_items, similar_items_columns=sim_items_columns, similarity_values=sim_vals, similar_items_y=sim_items_y, tolerance=PRICE_SIM_TOLLERANCE, house_fixed=house_fixed,
    real_y_form=real_y_form)

@app.route("/real-y/", methods=['POST'])
def do_send_real_y():
    form=RealYForm(request.form)
    
    if not form.validate(): # goes back
        session['real_y_form'] = encode_json(form.data)
        flash(u'Invalid price inserted', 'danger')
        #return redirect(request.referrer)
        return redirect(url_for('predict'))
    elif not 'x' in session: # goes home
        return redirect(url_for('index'))
    
    req_data = prepare_real_y_data(session['x'], form)
    
    if send_real_y(req_data):
        ## if ok, clean data from session
        if 'real_y_form' in session :
            session.pop('real_y_form')
        if 'house_form' in session :
            session.pop('house_form')
        if 'x' in session :
            session.pop('x')

        flash('Dati salvati correttamente', 'success')
        return redirect(url_for('index'))
    else:
        flash('Si &egrave; verificato un errore di comunicazione col server', 'danger')
        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True, host= '0.0.0.0', port=8084)
