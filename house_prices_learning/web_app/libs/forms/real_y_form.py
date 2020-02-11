from flask_wtf import FlaskForm as Form
from wtforms import DecimalField, SubmitField, validators #, StringField, BooleanField
 
class RealYForm(Form):
   real_y = DecimalField("Insert the real sell price (Euro):",
                [validators.InputRequired("Please insert the real price"), 
                 validators.NumberRange(min=0)],
                 render_kw={"placeholder": "> 0 e.g. 120500", "size": 10})
   submit = SubmitField("Send!")
