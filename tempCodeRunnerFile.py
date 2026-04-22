import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('model_laptop.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Prediksi Harga Laptop")

# Input user
processor = st.number_input("Processor Speed")
ram = st.number_input("RAM Size")
storage = st.number_input("Storage Capacity")
screen = st.number_input("Screen Size")
weight = st.number_input("Weight")

if st.button("Prediksi"):
    data = np.array([[processor, ram, storage, screen, weight]])
    data = scaler.transform(data)

    pred = model.predict(data)
    price = np.exp(pred)  # balik dari log

    st.success(f"Prediksi Harga: {price[0]:,.2f}")