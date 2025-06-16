import streamlit as st
import joblib
import numpy as np

model = joblib.load('model_2.pkl')
le_day = joblib.load('label_encoder_day.pkl')
le_season = joblib.load('label_encoder_season.pkl')
le_time = joblib.load('label_encoder_time.pkl')

st.title("Predikcija broja pacijenata po vremenskim varijablama")

day_options = list(le_day.classes_)
season_options = list(le_season.classes_)
time_options = list(le_time.classes_)

day_input = st.selectbox("Izaberi dan u sedmici:", day_options)
season_input = st.selectbox("Izaberi godišnje doba:", season_options)
time_input = st.selectbox("Izaberi dio dana:", time_options)

if st.button("Predvidi broj pacijenata"):
    day_enc = le_day.transform([day_input])[0]
    season_enc = le_season.transform([season_input])[0]
    time_enc = le_time.transform([time_input])[0]

    X_new = np.array([[day_enc, season_enc, time_enc]])

    prediction = model.predict(X_new)[0]

    st.success(f"Predviđeni broj pacijenata: {int(round(prediction))}")

