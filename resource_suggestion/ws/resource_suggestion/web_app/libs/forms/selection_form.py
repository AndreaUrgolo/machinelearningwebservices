from flask_wtf import FlaskForm as Form
from wtforms import SelectField, SubmitField, validators
import numpy as np 

class SelectionForm(Form):
    
    oc_choices = [(0, '---')] + list(zip(np.arange(1,len(ocs)+1), ocs))
    
    submit = SubmitField("Search")
    ocs_select = SelectField("Operative center:",
        [validators.Required("Please choose an operative center."), validators.NumberRange(min=1, max=len(ocs)+1)],
        choices=oc_choices,
        render_kw={"placeholder": "Choose an operative center"})
