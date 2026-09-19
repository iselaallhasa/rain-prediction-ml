
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_model():
    """Load the trained LightGBM model"""
    try:
        model = joblib.load('lightgbm_rain_model.pkl')
        feature_names = model.booster_.feature_name()
        return model, feature_names
    except FileNotFoundError:
        st.error("Model file 'lightgbm_rain_model.pkl' not found. Please make sure it's in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def engineer_features(df):
    df = df.copy()
    
  
    df['RainTodayBinary'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    
    
    df['TempDifference'] = df['MaxTemp'] - df['MinTemp']
    df['HumidityDifference'] = df['Humidity3pm'] - df['Humidity9am']
    df['PressureDifference'] = df['Pressure3pm'] - df['Pressure9am']
    df['WindSpeedInteraction'] = df['WindSpeed9am'] * df['WindSpeed3pm']
    df['WindGustRatio'] = df['WindGustSpeed'] / (df['WindSpeed3pm'] + 1)
    df['AvgTemp'] = (df['MinTemp'] + df['MaxTemp']) / 2
    df['AvgHumidity'] = (df['Humidity9am'] + df['Humidity3pm']) / 2
    df['AvgPressure'] = (df['Pressure9am'] + df['Pressure3pm']) / 2
    df['TempHumidityIndex'] = df['AvgTemp'] * df['AvgHumidity']
    df['PressureHumidityInteraction'] = df['AvgPressure'] * df['AvgHumidity']
    df['Rainfall_Log'] = np.log1p(df['Rainfall'])
    
    return df

def encode_categorical(df):
    """Encode categorical variables to numeric"""
    df = df.copy()
    
    wind_map = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}
    
    if 'WindGustDir' in df.columns:
        df['WindGustDir'] = df['WindGustDir'].map(wind_map).fillna(0)
    if 'WindDir9am' in df.columns:
        df['WindDir9am'] = df['WindDir9am'].map(wind_map).fillna(0)
    if 'WindDir3pm' in df.columns:
        df['WindDir3pm'] = df['WindDir3pm'].map(wind_map).fillna(0)
    
    # Location mapping (common locations)
    locations = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle',
                 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport',
                 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong',
                 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne',
                 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
                 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera',
                 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums',
                 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine']
    
    location_map = {loc: i for i, loc in enumerate(locations)}
    if 'Location' in df.columns:
        df['Location'] = df['Location'].map(location_map).fillna(0)
    
    # RainToday to binary
    if 'RainToday' in df.columns:
        df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    
    return df

def prepare_features(input_data, model_features):
    """Prepare input data for prediction"""
    
    df = pd.DataFrame([input_data])
    
    df = encode_categorical(df)
    
    df = engineer_features(df)
    
    
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0
    
  
    df = df[model_features]
    df = df.astype(float)
    
    return df

def predict_rain(model, model_features, input_data):
    """Make rain prediction"""
    

    prepared_data = prepare_features(input_data, model_features)
    

    probabilities = model.predict_proba(prepared_data)[0]
    
    prob_no_rain = probabilities[0]
    prob_rain = probabilities[1]
    

    prediction = 1 if prob_rain > 0.5 else 0
    
    return prediction, prob_rain, prob_no_rain

def get_weather_advice(probability, temp, humidity, rainfall, wind_speed):
  
    advice = []
    
    if probability >= 0.70:
        advice.append("High chance of rain! Don't forget your umbrella and raincoat.")
    elif probability >= 0.40:
        advice.append("Moderate chance of rain. Consider carrying an umbrella.")
    else:
        advice.append("Low chance of rain. Enjoy the weather!")
    
    if temp < 10:
        advice.append("It's quite cold - dress warmly!")
    elif temp > 35:
        advice.append("Hot weather - stay hydrated!")
    
    if humidity > 80:
        advice.append("High humidity - you might feel sticky.")
    
    if rainfall > 10:
        advice.append("Heavy rainfall expected - avoid outdoor activities.")
    
    if wind_speed > 50:
        advice.append("Strong winds - secure loose items outdoors.")
    
    return " ".join(advice)