

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from main import load_model, predict_rain, get_weather_advice

st.set_page_config(
    page_title="Rain Prediction System",
    page_icon="",
    layout="wide"
)

def apply_theme(theme):
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp { background-color: #0a0a2a; }
        .main-header {
            background: linear-gradient(135deg, #0a0a2a 0%, #1a1a4a 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            border: 1px solid #2a2a5a;
        }
        .main-header h1 { color: #ffffff !important; margin: 0; }
        .main-header p { color: #c0c0c0 !important; margin: 0.5rem 0 0 0; }
        .prediction-card {
            background: linear-gradient(135deg, #0d0d35 0%, #151545 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 1px solid #2a2a5a;
        }
        .prediction-card h3, .prediction-card p { color: #e0e0e0 !important; }
        .metric-card {
            background: linear-gradient(135deg, #0f0f3a 0%, #181848 100%);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #2a2a5a;
        }
        .metric-card h3 { color: #e0e0e0 !important; }
        .metric-card p { color: #ffffff !important; font-size: 2rem; font-weight: bold; margin: 0; }
        .stButton > button {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-weight: bold;
            border-radius: 8px;
            width: 100%;
        }
        h1, h2, h3 { color: #e0e0e0 !important; }
        .stSelectbox label, .stNumberInput label { color: #c0c0c0 !important; }
        .footer {
            text-align: center;
            padding: 1.5rem;
            margin-top: 2rem;
            border-top: 1px solid #2a2a5a;
            color: #a0a0a0;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #f5f5f0; }
        .main-header {
            background: linear-gradient(135deg, #1a3a6a 0%, #2a5a9a 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 { color: #ffffff !important; margin: 0; }
        .main-header p { color: #e0e0e0 !important; margin: 0.5rem 0 0 0; }
        .prediction-card {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #1a3a6a;
        }
        .prediction-card h3 { color: #1a3a6a !important; }
        .prediction-card p { color: #333333 !important; }
        .metric-card {
            background: #ffffff;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        .metric-card h3 { color: #1a3a6a !important; }
        .metric-card p { color: #1a3a6a !important; font-size: 2rem; font-weight: bold; margin: 0; }
        .stButton > button {
            background: linear-gradient(135deg, #1a3a6a 0%, #2a5a9a 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-weight: bold;
            border-radius: 8px;
            width: 100%;
        }
        h1, h2, h3 { color: #1a3a6a !important; }
        .stSelectbox label, .stNumberInput label { color: #333333 !important; }
        .footer {
            text-align: center;
            padding: 1.5rem;
            margin-top: 2rem;
            border-top: 1px solid #ddd;
            color: #666;
        }
        </style>
        """, unsafe_allow_html=True)

if 'theme' not in st.session_state:
    st.session_state.theme = "light"


model, model_features = load_model()


with st.sidebar:
    st.markdown("---")
    
    # Logo
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="width: 80px; height: 80px; background: linear-gradient(135deg, #1a3a6a, #2a5a9a); 
                    border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
            <span style="color: white; font-size: 36px; font-weight: bold;">CY</span>
        </div>
        <p style="margin-top: 0.5rem; font-weight: bold;">CY Tech University</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("Change Theme", use_container_width=True):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()
    
    st.markdown("---")
    
    with st.expander("About"):
        st.markdown("""
        LightGBM model predicts rain probability for tomorrow based on current weather conditions.
        """)

apply_theme(st.session_state.theme)


st.markdown("""
<div class="main-header">
    <h1>Rain Prediction System</h1>
    <p>Powered by LightGBM Machine Learning</p>
</div>
""", unsafe_allow_html=True)


with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Weather Parameters")
        
        location = st.selectbox(
            "Location",
            ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Canberra',
             'Hobart', 'Darwin', 'Albury', 'Cobar', 'CoffsHarbour', 'Newcastle',
             'WaggaWagga', 'Wollongong', 'Ballarat', 'Bendigo', 'Mildura']
        )
        
        col_temp1, col_temp2 = st.columns(2)
        with col_temp1:
            min_temp = st.number_input("Min Temperature (C)", value=15.0, step=0.5)
        with col_temp2:
            max_temp = st.number_input("Max Temperature (C)", value=25.0, step=0.5)
        
        col_hum1, col_hum2 = st.columns(2)
        with col_hum1:
            humidity_9am = st.slider("Humidity 9am (%)", 0, 100, 65)
        with col_hum2:
            humidity_3pm = st.slider("Humidity 3pm (%)", 0, 100, 50)
        
        rainfall = st.number_input("Rainfall (mm)", value=0.0, step=0.5)
        rain_today = st.selectbox("Rain Today?", ['No', 'Yes'])
    
    with col2:
        st.markdown("### Wind and Pressure")
        
        col_wind1, col_wind2 = st.columns(2)
        with col_wind1:
            wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", value=35, step=5, min_value=0)
            wind_speed_9am = st.number_input("Wind Speed 9am (km/h)", value=15, step=5, min_value=0)
        with col_wind2:
            wind_speed_3pm = st.number_input("Wind Speed 3pm (km/h)", value=20, step=5, min_value=0)
            wind_gust_dir = st.selectbox("Wind Gust Direction", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        
        col_press1, col_press2 = st.columns(2)
        with col_press1:
            pressure_9am = st.number_input("Pressure 9am (hPa)", value=1015.0, step=1.0)
        with col_press2:
            pressure_3pm = st.number_input("Pressure 3pm (hPa)", value=1012.0, step=1.0)
    
    with st.expander("Additional Parameters"):
        col_add1, col_add2, col_add3 = st.columns(3)
        
        with col_add1:
            cloud_9am = st.slider("Cloud 9am (oktas)", 0, 8, 3)
            cloud_3pm = st.slider("Cloud 3pm (oktas)", 0, 8, 3)
            evaporation = st.number_input("Evaporation (mm)", value=5.0, step=0.5)
        
        with col_add2:
            temp_9am = st.number_input("Temperature 9am (C)", value=18.0, step=0.5)
            temp_3pm = st.number_input("Temperature 3pm (C)", value=24.0, step=0.5)
            sunshine = st.number_input("Sunshine (hours)", value=8.0, step=0.5)
        
        with col_add3:
            wind_dir_9am = st.selectbox("Wind Direction 9am", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
            wind_dir_3pm = st.selectbox("Wind Direction 3pm", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    
    
    submitted = st.form_submit_button("PREDICT RAIN PROBABILITY", use_container_width=True)

# Make prediction
if submitted and model is not None:
    input_data = {
        'Location': location,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': evaporation,
        'Sunshine': sunshine,
        'WindGustDir': wind_gust_dir,
        'WindGustSpeed': wind_gust_speed,
        'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm,
        'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'Temp9am': temp_9am,
        'Temp3pm': temp_3pm,
        'RainToday': rain_today
    }
    
    with st.spinner("Calculating..."):
        prediction, prob_rain, prob_no_rain = predict_rain(model, model_features, input_data)
    
    if prediction is not None:
        st.markdown("---")
        st.markdown("## Prediction Results")
        
        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Rain Probability</h3>
                <p>{prob_rain:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            result_text = "RAIN EXPECTED" if prediction == 1 else "NO RAIN"
            result_color = "#dc2626" if prediction == 1 else "#10b981"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Prediction</h3>
                <p style="color: {result_color} !important;">{result_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            confidence = prob_rain if prediction == 1 else prob_no_rain
            st.markdown(f"""
            <div class="metric-card">
                <h3>Confidence</h3>
                <p>{confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        
        advice = get_weather_advice(prob_rain, max_temp, humidity_3pm, rainfall, wind_gust_speed)
        st.markdown(f"""
        <div class="prediction-card">
            <h3>Weather Advice</h3>
            <p>{advice}</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_rain * 100,
            title={'text': "Rain Probability", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1a3a6a"},
                'bgcolor': "white" if st.session_state.theme == "light" else "#1a1a4a",
                'steps': [
                    {'range': [0, 30], 'color': '#90EE90'},
                    {'range': [30, 70], 'color': '#FFD700'},
                    {'range': [70, 100], 'color': '#FF6B6B'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prob_rain * 100
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig, use_container_width=True)

elif submitted and model is None:
    st.error("Model not loaded. Please ensure 'lightgbm_rain_model.pkl' is in the same directory.")

st.markdown("""
<div class="footer">
    <p>Developed by: <strong>Youssef ALY | Isela Allhasa | Pietro Vincetin</strong></p>
    <p>CY Tech University - Rain Prediction System</p>
</div>
""", unsafe_allow_html=True)