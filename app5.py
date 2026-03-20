import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


model = joblib.load("fire_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="🔥 Fire Alert System", layout="wide")

st.title("🔥 Real-Time Fire Risk Prediction System")

menu = st.sidebar.selectbox("Menu", ["Home", "Live Prediction", "Dashboard"])


if "report" not in st.session_state:
    st.session_state.report = None


if menu == "Home":
    st.subheader("🌲 Forest Fire Prediction System")
    st.write("""
    This system predicts forest fire risk using:
    - 🌡 Temperature  
    - 💨 Wind Speed  
    - 💧 Humidity  
    - 🌧 Rainfall  

    Weather data is fetched from Open-Meteo API.
    """)


elif menu == "Live Prediction":

    st.subheader("📍 Enter City Name")
    city = st.text_input("Type City Name (Example: Howrah, Delhi, Mumbai)")

    if st.button("Generate Fire Risk Report"):

        try:
            
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
            geo_data = requests.get(geo_url).json()

            lat = geo_data['results'][0]['latitude']
            lon = geo_data['results'][0]['longitude']

            
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            weather_data = requests.get(weather_url).json()

            temp = weather_data['current_weather']['temperature']
            wind = weather_data['current_weather']['windspeed']
            RH = 40
            rain = 0

            
            input_data = np.array([[temp, RH, wind, rain]])
            input_scaled = scaler.transform(input_data)
            result = model.predict(input_scaled)[0]

            risk_status = "HIGH FIRE RISK 🚨" if result == 1 else "LOW FIRE RISK 🌿"

            
            st.session_state.report = {
                "city": city.upper(),
                "latitude": lat,
                "longitude": lon,
                "temperature": temp,
                "humidity": RH,
                "wind": wind,
                "rain": rain,
                "result": risk_status,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            
        
            map_data = pd.DataFrame({
                "lat": [lat],
                "lon": [lon]
            })
            st.map(map_data)

            
            st.markdown("## 📋 Fire Risk Prediction Report")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**City:** {city.upper()}")
                st.write(f"**Latitude:** {lat}")
                st.write(f"**Longitude:** {lon}")
                st.write(f"**Date & Time:** {st.session_state.report['time']}")

            with col2:
                st.write(f"🌡 **Temperature:** {temp} °C")
                st.write(f"💧 **Humidity:** {RH} %")
                st.write(f"💨 **Wind Speed:** {wind} km/h")
                st.write(f"🌧 **Rainfall:** {rain} mm")

            st.markdown("---")

            if result == 1:
                st.error(f"🔥 FINAL RESULT: {risk_status}")
            else:
                st.success(f"🔥 FINAL RESULT: {risk_status}")

        except:
            st.error("❌ City not found or API error. Please try again.")


elif menu == "Dashboard":

    st.subheader("📊 Fire Risk Dashboard")

    if st.session_state.report is None:
        st.warning("No report generated yet. Please generate a prediction first.")
    else:
        report = st.session_state.report

        
        map_data = pd.DataFrame({
            "lat": [report["latitude"]],
            "lon": [report["longitude"]]
        })
        st.map(map_data)

        st.markdown("## 📋 Last Prediction Report")

        st.write(f"**City:** {report['city']}")
        st.write(f"**Date & Time:** {report['time']}")

        st.write("### 🌤 Weather Data Used")
        st.write(f"🌡 Temperature: {report['temperature']} °C")
        st.write(f"💧 Humidity: {report['humidity']} %")
        st.write(f"💨 Wind Speed: {report['wind']} km/h")
        st.write(f"🌧 Rainfall: {report['rain']} mm")

        st.write("### 🔥 Fire Risk Result")
        if "HIGH" in report["result"]:
            st.error(report["result"])
        else:
            st.success(report["result"])

        st.info("Model: Random Forest | API: Open-Meteo")