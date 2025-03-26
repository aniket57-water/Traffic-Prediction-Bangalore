import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime

# Function to create neural network model
def create_neural_network(input_shape, output_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_shape)  # Output layer for multi-target regression
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )
    return model

# Load the preprocessor
preprocessor = joblib.load('preprocessor.pkl')

# Function to load models
def load_model(model_path):
    if model_path.endswith('.keras') or model_path.endswith('.h5'):
        return tf.keras.models.load_model(model_path)
    else:
        return joblib.load(model_path)

# Load the models
models = {
    'Gradient Boosting': load_model('models3/gradient_boosting.pkl'),
    'Support Vector Machine': load_model('models3/support_vector_machine.pkl'),
    'Neural Network': load_model('models3/neural_network.keras')
}

# Streamlit app
st.title('Bangalore Traffic Prediction System')

st.markdown("""
This application predicts traffic conditions in Bangalore based on various parameters.
Select your inputs below and click 'Predict' to see the results.
""")

# Sidebar for model selection
st.sidebar.header('Model Selection')
selected_model = st.sidebar.selectbox('Choose a prediction model:', list(models.keys()))

# Main input form
with st.form("traffic_prediction_form"):
    st.header('Input Parameters')
    
    col1, col2 = st.columns(2)
    
    with col1:
        area_name = st.selectbox('Area Name', [
            'Indiranagar', 'Whitefield', 'Koramangala', 'M.G. Road', 
            'Jayanagar', 'Hebbal', 'Yeshwanthpur', 'Electronic City'
        ])
        
        road_name = st.text_input('Road/Intersection Name', '100 Feet Road')
        
        date = st.date_input('Date', datetime.today())
        day = date.day
        month = date.month
        day_of_week = date.weekday()  # Monday=0, Sunday=6
        
        road_capacity = st.slider('Road Capacity Utilization (%)', 0, 100, 50)
        
    with col2:
        incidents = st.slider('Incident Reports (count)', 0, 10, 0)
        public_transport = st.slider('Public Transport Usage (%)', 0, 100, 50)
        signal_compliance = st.slider('Traffic Signal Compliance (%)', 0, 100, 80)
        parking_usage = st.slider('Parking Usage (%)', 0, 100, 50)
        
    col3, col4 = st.columns(2)
    
    with col3:
        pedestrian_count = st.slider('Pedestrian and Cyclist Count', 0, 300, 100)
        
    with col4:
        weather = st.selectbox('Weather Conditions', [
            'Clear', 'Overcast', 'Rain', 'Fog', 'Windy'
        ])
        
    roadwork = st.radio('Roadwork and Construction Activity', ['Yes', 'No'])
    
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'Area Name': [area_name],
        'Road/Intersection Name': [road_name],
        'Road Capacity Utilization': [road_capacity],
        'Incident Reports': [incidents],
        'Public Transport Usage': [public_transport],
        'Traffic Signal Compliance': [signal_compliance],
        'Parking Usage': [parking_usage],
        'Pedestrian and Cyclist Count': [pedestrian_count],
        'Weather Conditions': [weather],
        'Roadwork and Construction Activity': [roadwork],
        'Day': [day],
        'Month': [month],
        'Day_of_week': [day_of_week]
    })
    
    # Preprocess the input
    input_preprocessed = preprocessor.transform(input_data)
    
    # Make prediction
    model = models[selected_model]
    
    # Handle prediction for different model types
    if selected_model == 'Neural Network':
        prediction = model.predict(input_preprocessed)
    else:
        prediction = model.predict(input_preprocessed)
    
    # Display results
    st.header('Prediction Results')
    
    # Handle different prediction formats
    if isinstance(prediction, np.ndarray):
        if prediction.ndim == 1:
            results = {
                'Traffic Volume': int(prediction[0]),
                'Average Speed': round(prediction[1], 1),
                'Congestion Level': round(prediction[2], 1),
                'Travel Time Index': round(prediction[3], 2)
            }
        elif prediction.ndim == 2:
            results = {
                'Traffic Volume': int(prediction[0][0]),
                'Average Speed': round(prediction[0][1], 1),
                'Congestion Level': round(prediction[0][2], 1),
                'Travel Time Index': round(prediction[0][3], 2)
            }
    else:
        # For Keras model
        results = {
            'Traffic Volume': int(prediction[0][0]),
            'Average Speed': round(prediction[0][1], 1),
            'Congestion Level': round(prediction[0][2], 1),
            'Travel Time Index': round(prediction[0][3], 2)
        }
    
    st.subheader(f"Using {selected_model} Model:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Traffic Volume", f"{results['Traffic Volume']} vehicles")
        st.metric("Average Speed", f"{results['Average Speed']} km/h")
        
    with col2:
        st.metric("Congestion Level", f"{results['Congestion Level']}%")
        st.metric("Travel Time Index", results['Travel Time Index'])
    
    # Interpretation
    st.subheader("Interpretation:")
    
    congestion_level = results['Congestion Level']
    if congestion_level < 30:
        st.success("Low congestion - Traffic is flowing smoothly")
    elif 30 <= congestion_level < 70:
        st.warning("Moderate congestion - Expect some delays")
    else:
        st.error("High congestion - Significant delays expected")

# Add some information about the dataset and models
st.sidebar.header("About")
st.sidebar.info("""
This application uses machine learning models trained on Bangalore traffic data to predict:
- Traffic Volume
- Average Speed
- Congestion Level
- Travel Time Index
""")

st.sidebar.header("Model Performance")
st.sidebar.write("""
Models are evaluated based on their Mean Squared Error (MSE):
- Lower MSE indicates better performance
""")