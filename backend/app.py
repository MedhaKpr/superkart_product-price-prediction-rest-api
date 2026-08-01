# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API
#from flask_cors import CORS

# Initialize the Flask application
superkart_predictor_api = Flask("SuperKart Price Predictor")
#CORS(superkart_predictor_api)

#print("Flask app 'superkart_predictor_api' initialized and CORS enabled.")

# Load the trained machine learning model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page (GET request)
@superkart_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Products Price Prediction API!"

# Define an endpoint for single property prediction (POST request)
@superkart_predictor_api.post('/v1/product')
def predict_product_price():
    """
    This function handles POST requests to the '/v1/product' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted product price as a JSON response.
    """
    # Get the JSON data from the request body
    property_data = request.get_json()

     # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': property_data['Product_Weight'],
        'Product_Sugar_Content': property_data['Product_Sugar_Content'],
        'Product_Allocated_Area': property_data['Product_Allocated_Area'],
        'Product_MRP': property_data['Product_MRP'],
        'Store_Size': property_data['Store_Size'],
        'Store_Location_City_Type': property_data['Store_Location_City_Type'],
        'Store_Type': property_data['Store_Type'],
        'Store_Age_Years': property_data['Store_Age_Years'],
        'Product_Type_Category': property_data.get('Product_Type_Category', 'Other'),
        'Product_Id_char': property_data['Product_Id_char']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction (get log_price)
    predicted_log_price = model.predict(input_data)[0]

    # Calculate actual price
    predicted_price = np.exp(predicted_log_price)

    # Convert predicted_price to Python float
    predicted_price = round(float(predicted_price), 2)

    # Return the actual price
    return jsonify({'Predicted Price (in dollars)': predicted_price})


# Define an endpoint for batch prediction (POST request)
@superkart_predictor_api.post('/v1/batchproducts')
def predict_batch_products_price():
    """
    This function handles POST requests to the '/v1/batchproducts' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted product prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame (get log_prices)
    predicted_log_prices = model.predict(input_data).tolist()

    # Calculate actual prices
    predicted_prices = [round(float(np.exp(log_price)), 2) for log_price in predicted_log_prices]

    # Create a dictionary of predictions with property IDs as keys
    property_ids = input_data.index.tolist()  # Fallback to index if 'id' is missing
    if 'Product_Id' in input_data.columns:
        property_ids = input_data['Product_Id'].tolist()
    
    output_dict = dict(zip(property_ids, predicted_prices))

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_predictor_api.run(debug=True, host='0.0.0.0', port=7860)
