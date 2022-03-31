# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

# Load ML model
log_model = pickle.load(open('logreg_model.pkl', 'rb'))

# Create application
app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('index.html')

# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def predict():
    
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    pred = log_model.predict(final_features)
    
    return render_template('index.html', str_msg='The predicted fish type is {}'.format(pred))
   

if __name__ == '__main__':
#Run the application
    app.run()
