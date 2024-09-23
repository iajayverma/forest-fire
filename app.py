from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the Ridge Regressor model and Standard Scaler from pickle files
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/sacler.pkl', 'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction page
@app.route('/predictdata/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temperature = float(request.form['Temperature'])
        rh = float(request.form['RH'])
        ws = float(request.form['WS'])
        rain = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])
        classes = float(request.form['Classes'])
        region = float(request.form['Region'])
        input_data = [
            temperature, rh, ws, rain, ffmc, dmc, isi, classes, region
        ]
        new_scaled_data=standard_scaler.transform([input_data])

        result = ridge_model.predict(new_scaled_data)

        return render_template('home.html', result=round(result[0],2))


        
    else:
        return render_template('home.html')




      
    

    return render_template('home.html')

# Main function to run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)
