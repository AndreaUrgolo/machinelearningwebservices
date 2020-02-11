import datetime
import pandas as pd
from flask import Flask, flash, render_template, request, session, abort, jsonify, make_response, redirect, url_for
from flask_bootstrap import Bootstrap
import requests

import platform
import multiprocessing


""" from our libraries """

from config import *

from libs.utils.json_utils import *
from libs.utils.flask_utils import *

# FORMS

from libs.forms.selection_form import SelectionForm


app = Flask(__name__)
app.config.update(APP_CONFIG)
Bootstrap(app)


@app.route("/")
def index():
    ocs = parse_json(requests.get(APP_CONFIG['WS_URL'] + APP_CONFIG['OPERATIVE_CENTERS_REQ']).content)
    form=SelectionForm()
    return render_template('index.html', form=form, ocs=ocs)


if __name__ == "__main__":
    if platform.system() != 'Windows':
        multiprocessing.set_start_method('forkserver')
    app.run(debug=True, host='0.0.0.0', port=8087, threaded=True, use_reloader=False)
