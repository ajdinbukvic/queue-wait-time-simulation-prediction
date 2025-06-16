import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open('model_1.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("Predikcija vremena čekanja na osnovu zakazanih termina")

uploaded_file = st.file_uploader("Upload CSV fajl s ulaznim podacima", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Učitani podaci:")
    st.dataframe(data.head())

    features = [
        'Hospital ID', 'Region', 'Day of Week', 'Season', 'Time of Day',
        'Urgency Level', 'Nurse-to-Patient Ratio', 'Specialist Availability', 'Facility Size (Beds)'
    ]

    missing_cols = [col for col in features if col not in data.columns]
    if missing_cols:
        st.error(f"Nedostaju kolone u CSV-u: {missing_cols}")
    else:
        for col in ['Hospital ID', 'Region', 'Day of Week', 'Season', 'Time of Day', 'Urgency Level']:
            data[col] = data[col].astype('category').cat.codes

        X = data[features]

        preds = model.predict(X)
        data['Predicted Total Wait Time (min)'] = preds

        st.write("Rezultati predikcije:")
        st.dataframe(data)

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Preuzmi CSV sa predikcijama",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )