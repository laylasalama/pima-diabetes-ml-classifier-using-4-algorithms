import streamlit as st
import joblib
import pandas as pd
import numpy as np

# configure the browser tab title/icon and page layout
st.set_page_config(page_title='Diabetes Risk Predictor', page_icon='🩺', layout='centered')

# load the trained models, imputer, clip bounds, and scaler saved from the notebook
artifact = joblib.load('diabetes_model.pkl')

# name, default, min, max for each input slider
FIELDS = [
    ('Pregnancies', 1, 0, 20),
    ('Glucose', 120, 0, 300),
    ('BloodPressure', 70, 0, 150),
    ('SkinThickness', 20, 0, 100),
    ('Insulin', 80, 0, 1000),
    ('BMI', 25.0, 0, 70),
    ('DiabetesPedigreeFunction', 0.5, 0.0, 3.0),
    ('Age', 30, 0, 120),
]
INT_FIELDS = {'Pregnancies', 'Age'}
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# rebuild the exact preprocessing pipeline used in training: raw input, impute, scale, predict
def predict_with(model_name, values):
    raw = pd.DataFrame([values])[artifact['raw_columns']]
    raw[ZERO_AS_MISSING] = raw[ZERO_AS_MISSING].replace(0, np.nan)
    imputed = pd.DataFrame(artifact['imputer'].transform(raw), columns=artifact['raw_columns'])
    scaled = artifact['scaler'].transform(imputed)
    model = artifact['models'][model_name]
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]
    return prediction, probability

st.title('Diabetes Risk Predictor')

# lets the user pick which of the four trained models to run the prediction with
model_name = st.selectbox('Choose a model', list(artifact['models']))

# render one slider per feature, collecting user input into a dict for prediction
values = {}
for name, default, min_val, max_val in FIELDS:
    if name in INT_FIELDS:
        values[name] = st.slider(name, int(min_val), int(max_val), int(default))
    else:
        values[name] = st.slider(name, float(min_val), float(max_val), float(default), step=0.1)

# run the prediction and display the result only when the button is clicked
if st.button('Predict', type='primary', use_container_width=True):
    prediction, probability = predict_with(model_name, values)
    if prediction == 1:
        st.error(f'Higher risk of diabetes  ({probability:.1%})')
    else:
        st.success(f'Lower risk of diabetes  ({probability:.1%})')
    st.caption('This is a model estimate, not a medical diagnosis.')