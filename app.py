from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Define the paths to the model file
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")

# Load the trained model pipeline
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("Error: The model file was not found. Please run model.py first.")
    exit()

@app.route('/')
def home():
    """Renders the home page with the house price prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the form.
    
    The function collects form data, converts it into a pandas DataFrame,
    and uses the loaded Gradient Boosting model to predict the house price.
    The result is then rendered on the results page.
    """
    # Get form data from the request
    form_data = request.form.to_dict()

    # Convert numeric values to the correct type
    try:
        square_footage = float(form_data['SquareFootage'])
        bedrooms = int(form_data['Bedrooms'])
        bathrooms = float(form_data['Bathrooms'])
        year_built = int(form_data['YearBuilt'])
    except ValueError:
        return render_template('result.html', prediction="Error: Invalid input for numeric fields. Please enter numbers.")

    # Create a DataFrame with the new data, ensuring the column order matches the training data
    features = pd.DataFrame({
        'SquareFootage': [square_footage],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'YearBuilt': [year_built],
        'Location': [form_data['Location']]
    })

    # Make a prediction using the loaded model
    prediction = model.predict(features)[0]
    
    # Format the prediction as a currency string
    prediction_formatted = f"${prediction:,.2f}"

    # Return the prediction result
    return render_template('result.html', prediction=prediction_formatted)

if __name__ == '__main__':
    # You can run the app with `python app.py` and access it at http://127.0.0.1:5000
    app.run(debug=True)
