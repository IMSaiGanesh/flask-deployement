from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Extract input data from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    bmi = int(request.form['bmi'])
    children = int(request.form['children'])
    smoker =int( request.form['smoker'])
    region =int( request.form['region'])

    # Perform data preprocessing (if necessary)
    # For example, encode categorical variables and scale numerical features

    # Make predictions using the model
    prediction = model.predict([[age, sex, bmi, children, smoker, region]])[0]

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction})
if __name__ == '__main__':
    app.run(debug=True)
