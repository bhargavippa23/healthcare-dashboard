
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_model.pkl')

# Function to get the full feature set
def get_full_feature_set():
    # These are the exact columns used during model training
    feature_columns = [
        'Age', 'Billing Amount', 'Length of Stay',
        'Admission Type', 'Age_Bin_Adult', 'Age_Bin_Elderly',
        'Age_Bin_Senior', 'Age_Bin_Young Adult',
        # Add all other one-hot encoded features here
    ]
    return feature_columns

# Function to create the input dataframe
def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 30)
    billing_amount = st.sidebar.slider('Billing Amount', 1000, 50000, 15000)
    length_of_stay = st.sidebar.slider('Length of Stay', 1, 30, 5)
    
    # Add more features as necessary
    data = {
        'Age': age,
        'Billing Amount': billing_amount,
        'Length of Stay': length_of_stay,
        # Any other features you need to match the training set
    }
    
    # Convert the input into a DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Get the full feature set and ensure all required columns are present
    full_feature_set = get_full_feature_set()
    
    # Add missing columns with zeros
    for col in full_feature_set:
        if col not in features.columns:
            features[col] = 0
    
    # Reorder the columns to match the training set
    features = features[full_feature_set]
    
    return features

# Create the input data
input_df = user_input_features()

# Display user inputs
st.subheader("User Input Parameters")
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display the prediction
st.subheader("Prediction")
test_result = ['Normal', 'Abnormal', 'Inconclusive']  # Adjust based on your target variable
st.write(f'The predicted test result is: {test_result[prediction[0]]}')

st.subheader("Prediction Probability")
st.write(prediction_proba)

# Feature importance (if applicable)
if hasattr(model, 'feature_importances_'):
    st.subheader("Feature Importances")
    feature_importances = pd.Series(model.feature_importances_, index=input_df.columns)
    st.bar_chart(feature_importances)
