import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load the dataset
df = pd.read_csv('Bangalore Traffic data.csv')

# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Day_of_week'] = df['Date'].dt.dayofweek
df = df.drop('Date', axis=1)

# Define target variables
targets = ['Traffic Volume', 'Average Speed', 'Congestion Level', 'Travel Time Index']

# Define features
features = df.drop(targets, axis=1).columns

# Split into features and targets
X = df[features]
y = df[targets]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = ['Road Capacity Utilization', 'Incident Reports', 'Public Transport Usage', 
                   'Traffic Signal Compliance', 'Parking Usage', 'Pedestrian and Cyclist Count', 
                   'Day', 'Month', 'Day_of_week']
categorical_features = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 
                       'Roadwork and Construction Activity']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Function to create neural network model
def create_neural_network(input_shape, output_shape):
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

# Initialize models
models = {}

# Random Forest with MultiOutputRegressor
models['Random Forest'] = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))

# Gradient Boosting with MultiOutputRegressor
models['Gradient Boosting'] = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))

# Support Vector Machine with MultiOutputRegressor
models['Support Vector Machine'] = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))

# Multioutput MLP 
models['Multi-layer Perceptron'] = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42))

# Neural Network 
# Create and train neural network model separately
neural_model = create_neural_network(X_train_preprocessed.shape[1], len(targets))
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
neural_model.fit(X_train_preprocessed, y_train, 
                 validation_split=0.2,
                 epochs=50,
                 batch_size=32,
                 callbacks=[early_stopping])
models['Neural Network'] = neural_model

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    
    if name != 'Neural Network':
        model.fit(X_train_preprocessed, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_preprocessed)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'mse': mse,
        'r2': r2
    }
    
    print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")

# Save the best model and preprocessing pipeline
best_model_name = min(results, key=lambda x: results[x]['mse'])
best_model = results[best_model_name]['model']

print(f"\nBest model: {best_model_name} with MSE: {results[best_model_name]['mse']:.2f}")

# Save the model and preprocessor
import os

# Ensure models directory exists
os.makedirs('models3', exist_ok=True)

if best_model_name == 'Neural Network':
    # Save Keras model
    best_model.save('models3/neural_network.keras')
else:
    joblib.dump(best_model, 'models3/neural_network.pkl')

# Save preprocessor
joblib.dump(preprocessor, 'preprocessor.pkl')

# Save all models for comparison in the GUI
for name, model in models.items():
    if name == 'Neural Network':
        model.save(f'models3/{name.lower().replace(" ", "_")}.keras')
    else:
        joblib.dump(model, f'models3/{name.lower().replace(" ", "_")}.pkl')

print("Models and preprocessor saved successfully.")