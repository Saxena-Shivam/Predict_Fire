import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import ridge regressor and standard scaler
ridge_model = pickle.load(open('model/ridge.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            fields = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
            values = [float(request.form.get(f, 0)) for f in fields]
            arr = np.array(values).reshape(1, -1)
            arr_scaled = scaler.transform(arr)
            pred = ridge_model.predict(arr_scaled)
            result = float(pred[0])
        except Exception as e:
            result = f'Error: {e}'
        return render_template('home.html', result=result)
    else:
        return render_template('home.html')
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')