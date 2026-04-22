import streamlit as st
import numpy as np
import joblib

model = joblib.load('model_laptop.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Prediksi Harga Laptop")

# Input user
processor = st.number_input("Processor Speed (GHz)")
ram = st.number_input("RAM Size (GB)")
storage = st.number_input("Storage Capacity (GB)")
screen = st.number_input("Screen Size (inch)")
weight = st.number_input("Weight (kg)")

# 🔥 TAMBAHAN BRAND
brand = st.selectbox("Brand", ["Asus", "Dell", "HP", "Lenovo"])

if st.button("Prediksi"):

    # One-hot encoding manual
    brand_asus = 1 if brand == "Asus" else 0
    brand_dell = 1 if brand == "Dell" else 0
    brand_hp = 1 if brand == "HP" else 0
    brand_lenovo = 1 if brand == "Lenovo" else 0

    # 🔥 HARUS URUT SAMA DENGAN TRAINING
    data = np.array([[processor, ram, storage, screen, weight,
                      brand_asus, brand_dell, brand_hp, brand_lenovo]])

    data = scaler.transform(data)

    pred = model.predict(data)
    price = np.exp(pred)

    st.success(f"Prediksi Harga: {price[0]:,.2f}")