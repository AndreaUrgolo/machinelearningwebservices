from flask_wtf import FlaskForm as Form
from wtforms import IntegerField, DecimalField, SubmitField, validators #, StringField, BooleanField 

class HouseForm(Form):
   labels = {'CRIM':"Crime rate (%):", 'INDUS':"Industries rate (%):", 'RM':"Rooms:", 
             'AGE':"Aged rate (%):", 'DIS':"Employ Centre (miles):", 'TAX':"Property tax (for 50.000 Euro):", 
             'PTRATIO':"Pupil/teacher ratio:", 'LSTAT':"Lower status (%):"}
   # Form fields '##' for ignored fields
   #1. CRIM      per capita crime rate by town   
   crim = DecimalField("Crime rate (%): [*]",
                [validators.InputRequired("Please insert a crime rate."),
                 validators.NumberRange(min=0, max=100)],
                 render_kw={"placeholder": "0-100% e.g. 5.3", "size": 12})
   ##2. ZN        proportion of residential land zoned for lots over 
                 #25,000 sq.ft.
                 
   #3. INDUS     proportion of non-retail business acres per town
   indus = DecimalField("Industries rate (%):",
                [#validators.InputRequired("Please insert a industries rate."), 
                 validators.optional(),
                 validators.NumberRange(min=0, max=100)],
                 render_kw={"placeholder": "0-100% e.g. 10.5", "size": 12})
   ##4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
   #chas = BooleanField("Near a river")
      
   ##5. NOX       nitric oxides concentration (parts per 10 million)
    
   #6. RM        average number of rooms per dwelling
   rm = IntegerField("Rooms: [*]",
                [validators.InputRequired("Please insert the number of rooms."), validators.NumberRange(min=0)],
                 render_kw={"placeholder": ">0 e.g. 6", "size": 12})
   
   #7. AGE       proportion of owner-occupied units built prior to 1940
   age = DecimalField("Aged rate (%):",
                [#validators.InputRequired("Please insert the aged rate."), 
                 validators.optional(),
                 validators.NumberRange(min=0, max=100)],
                 render_kw={"placeholder": "0-100% Old units (built before 1940). e.g. 10.7","size": 26}) 
   
   
   #8. DIS       distances from employment centre
   dis = DecimalField("Employ Centre (km):",
                [#validators.InputRequired("Please insert the distance."), 
                 validators.optional(),
                 validators.NumberRange(min=0)],
                 render_kw={"placeholder": ">0 e.g. 5", "size": 12})
   
   ##9. RAD      index of accessibility to radial highways
   
   #10. TAX      full-value property-tax rate per $10,000
   tax = DecimalField("Property tax (for 50.000 Euro):",
                [#validators.InputRequired("Please insert the property tax."), 
                 validators.optional(),
                 validators.NumberRange(min=0, max=50000)],
                 render_kw={"placeholder": "0-50000 e.g. 400.00", "size": 12})
   
   #11. PTRATIO  pupil-teacher ratio by town
   ptratio = DecimalField("Pupil/teacher ratio:",
                [#validators.InputRequired("Please insert the pupil rate."),
                 validators.optional(),
                 validators.NumberRange(min=0)],
                 render_kw={"placeholder": ">0 e.g. 17.91", "size": 12})
   
   ##12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
   
   #13. LSTAT    % lower status of the population
   lstat =  DecimalField("Lower status (%): [*]",
                [validators.InputRequired("Please insert the lower status rate."), 
                 validators.NumberRange(min=0, max=100)],
                 render_kw={"placeholder": "0-100% e.g. 12.31", "size": 12})

   submit = SubmitField("Predict!")
   submit_sim = SubmitField("Get Similar!") 
