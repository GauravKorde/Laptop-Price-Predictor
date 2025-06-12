import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
old_df = pickle.load(open('old_df', 'rb'))

# Title
st.title("💻 Laptop Price Predictor")
st.markdown("Use this ML-powered app to predict the price of a laptop based on its specifications.")

# Sidebar
st.sidebar.header("📘 About the App")
st.sidebar.markdown("""
This app uses a machine learning model trained on real-world laptop data to predict laptop prices.
- Developed by: **Your Name**
- Model Accuracy: ~90% (R² Score)
- Data Source: Kaggle (Laptop Price Dataset)
""")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('🏢 Brand', df['Company'].unique())
    type = st.selectbox('💼 Type', df['TypeName'].unique())
    ram = st.selectbox('🧠 RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('⚖️ Weight of the Laptop (kg)', min_value=0.5, max_value=5.0, step=0.1)
    touchscreen = st.selectbox('📱 Touchscreen', ['No', 'Yes'])

with col2:
    ips = st.selectbox('🖥️ IPS Display', ['No', 'Yes'])
    screen_size = st.slider('📐 Screen Size (inches)', 10.0, 18.0, 13.3)
    resolution = st.selectbox('🔍 Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160',
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])
    cpu = st.selectbox('🧮 CPU Brand', df['Cpu brand'].unique())
    hdd = st.selectbox('💾 HDD (GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('🚀 SSD (GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('🎮 GPU Brand', df['Gpu brand'].unique())
    os = st.selectbox('🧑‍💻 Operating System', df['os'].unique())

# Predict button
if st.button('📊 Predict Price'):

    # Convert Yes/No to binary
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # Resolution to PPI
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

    # Input validation
    if weight <= 0:
        st.error("⚠️ Weight must be a positive number.")
    else:
        # Prepare input for model
        query_dict = {
            'Company': [company],
            'TypeName': [type],
            'Ram': [ram],
            'Weight': [weight],
            'Touchscreen': [touchscreen_val],
            'Ips': [ips_val],
            'ppi': [ppi],
            'Cpu brand': [cpu],
            'HDD': [hdd],
            'SSD': [ssd],
            'Gpu brand': [gpu],
            'os': [os]
        }

        query_df = pd.DataFrame(query_dict)

        # Prediction with spinner
        with st.spinner('⏳ Predicting price...'):
            prediction = pipe.predict(query_df)
            predicted_price = int(np.exp(prediction[0]))  # Because model trained on log(price)
            st.success(f"💰 The predicted price is **₹ {predicted_price}**")


st.markdown("#### 🔎 Similar Laptops from Dataset")
similar = old_df[(df['Company'] == company) & (old_df['TypeName'] == type) & (ram in old_df['Ram'] ) & (cpu in old_df["Cpu"]) & (gpu in old_df["Gpu"])]
st.dataframe(similar.head(5))
