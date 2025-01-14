from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf

# Define the custom MSE function
def mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model
model_path = "./model.h5"  # Adjust the path if necessary
model = load_model(model_path, custom_objects={"mse": mse})

# Load the saved scalers
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Define the request body schema
class PredictionRequest(BaseModel):
    data: list  # Input data as a list of features

# API Endpoint: Health check
@app.get("/")
def health_check():
    return {"message": "ML Model API is running!"}

# API Endpoint: Predict
@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        # Validate input data
        if not request.data or len(request.data) == 0:
            raise HTTPException(status_code=400, detail="Input data cannot be empty")

        # Convert input data to numpy array and reshape it for the model
        input_data = np.array(request.data).reshape(1, 1, -1)

        # Scale the input data (features)
        input_data_scaled = scaler_X.transform(input_data.reshape(1, -1)).reshape(1, 1, -1)

        # Make the prediction
        scaled_prediction = model.predict(input_data_scaled)

        # Inverse transform the prediction back to the original scale (target variable)
        original_prediction = scaler_y.inverse_transform(scaled_prediction)

        # Convert the prediction to a regular Python float
        result = float(original_prediction[0][0])

        # Return the prediction as a float
        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the prediction: {str(e)}")
