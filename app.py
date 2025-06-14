import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import pandas as pd

# Load model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
old_df = pickle.load(open('old_df.pkl', 'rb'))

# Title
st.title("💻 Laptop Price Predictor")
with st.expander("💡 What Specs Should You Choose? (Click to expand)"):
    st.markdown("""
    #### 👨‍💻 Programming & Coding (Python, Java, etc.)
    - **Processor:** Intel i5 / AMD Ryzen 5 or better  
    - **RAM:** 8GB minimum  
    - **Storage:** SSD preferred  
    - **GPU:** Not necessary

    #### 📚 Online Classes & Notes
    - **Processor:** Intel i3 or above  
    - **RAM:** 4GB minimum  
    - **Storage:** Any  
    - **GPU:** Integrated GPU is fine

    #### 🎮 Gaming
    - **Processor:** i5/i7 or Ryzen 5/7  
    - **RAM:** 16GB  
    - **GPU:** NVIDIA GTX/RTX or AMD Radeon  
    - **Storage:** SSD + optional HDD

    #### 🧠 Machine Learning / Deep Learning
    - **Processor:** i7 or Ryzen 7+  
    - **RAM:** 16GB+  
    - **GPU:** **NVIDIA GPU (RTX series preferred)**  
    - **Storage:** SSD for fast data access

    #### 🎨 Design / Video Editing
    - **Processor:** i7 or Ryzen 7  
    - **RAM:** 16GB+  
    - **GPU:** Dedicated GPU  
    - **Display:** High resolution, IPS panel

    ---

    If you're confused, look for laptops with:
    - **At least 8GB RAM**
    - **SSD over HDD**
    - **Decent CPU (i5 or Ryzen 5 minimum)**
    - **NVIDIA GPU if you're doing ML or gaming**
    """)

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('🏢 Brand', df['Company'].unique())
    type = st.selectbox('💼 Type', df['TypeName'].unique())
    ram = st.selectbox('🧠 RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('⚖️ Weight of the Laptop (kg)', min_value=0.5, max_value=5.0, step=0.1)
    touchscreen = st.selectbox('📱 Touchscreen', ['No', 'Yes'])
    screen_size = st.slider('📐 Screen Size (inches)', 10.0, 18.0, 13.3)
    resolution = st.selectbox('🔍 Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160',
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])

with col2:
    cpu = st.selectbox('🧮 CPU Brand', df['Cpu brand'].unique())
    hdd = st.selectbox('💾 HDD (GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('🚀 SSD (GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('🎮 GPU Brand', df['Gpu brand'].unique())
    ips = st.selectbox('🖥️ IPS Display', ['No', 'Yes'])
    os = st.selectbox('🧑‍💻 Operating System', df['os'].unique())

with st.sidebar:
    st.image("student_python.jpg", width=180, caption="Python-powered student 💻")
    st.title("🧠 Laptop Price Guide")

    st.markdown("### 🚀 How to Use This Model")
    st.markdown("""
    1. Select laptop specs from the dropdowns  
    2. Click on **📊 Predict Price**  
    3. View the **predicted price in ₹**  
    4. Copy the **search prompt** to find similar laptops online  
    5. Scroll down to see **similar laptops** from the dataset
    """)



    st.markdown("### 💡 Smart Tips:")
    st.info("""
    - Use **SSD over HDD** for faster performance  
    - Remove **Touchscreen / IPS** to reduce cost  
    - Use the prompt to explore similar laptops online  
    """)

    st.markdown("### 🔍 What This App Offers")
    st.markdown("""
    - Predicts laptop price using ML  
    - Generates a ready-to-copy search prompt  
    - Shows similar laptops from dataset  
    - Links to Amazon & Flipkart  
    """)

    st.markdown("---")
    st.caption("🔧 Created by: [Gaurav Korde]")
    st.caption("📅 Version: 1.0 | Powered by Random Forest")

# Predict button
if st.button('📊 Predict Price'):

    # Convert Yes/No to binary
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # Resolution to PPI
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2) ** 0.5) / float(screen_size)

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
            st.markdown(f"""
            <div style="background-color:#dff0d8;padding:20px;border-radius:10px">
                <h2 style="color:green;text-align:center;">💰 Predicted Price: ₹ {predicted_price}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Smart search prompt
        st.markdown("<br><br>", unsafe_allow_html=True)  # Adds vertical space
        st.markdown(
            "<div style='margin-top:20px; font-size:18px; font-weight:bold;'>🧭 This search prompt is provided to help you find laptops with similar specifications online.</div>",
            unsafe_allow_html=True
        )

        search_prompt = (
            f"{company} {type} laptop, {ram}GB RAM, {cpu} processor, "
            f"{gpu} graphics, {ssd}GB SSD"
            f"{', ' + str(hdd) + 'GB HDD' if hdd > 0 else ''}, "
            f"{os}, {screen_size:.1f}\" screen, "
            f"{'Touchscreen' if touchscreen_val else 'Non-Touch'}, "
            f"{'IPS display' if ips_val else 'Non-IPS display'}"
        )

        # Display the search prompt as code
        st.code(search_prompt, language='text')

        # Reliable copy to clipboard button using navigator.clipboard
        components.html(f"""
            <div>
                <button id="copy-btn" style="
                    background-color:#4CAF50;
                    color:white;
                    border:none;
                    padding:10px 20px;
                    border-radius:5px;
                    cursor:pointer;
                    font-weight:bold;
                ">📎 Copy Prompt</button>
                <p id="status" style="color:green; font-weight:bold;"></p>
            </div>

            <script>
                const copyBtn = document.getElementById("copy-btn");
                const status = document.getElementById("status");

                copyBtn.onclick = async function() {{
                    try {{
                        await navigator.clipboard.writeText(`{search_prompt}`);
                        status.innerText = "✅ Copied to clipboard!";
                        setTimeout(() => status.innerText = "", 2000);
                    }} catch (err) {{
                        status.innerText = "❌ Failed to copy";
                    }}
                }}
            </script>
        """, height=120)

        # Optional direct links
        amazon_url = f"https://www.amazon.in/s?k={search_prompt.replace(' ', '+')}"
        flipkart_url = f"https://www.flipkart.com/search?q={search_prompt.replace(' ', '+')}"
        st.markdown(f"[🔗 Search on Amazon]({amazon_url})")
        st.markdown(f"[🔗 Search on Flipkart]({flipkart_url})")

# Similar laptops from dataset
st.markdown("#### 🔎 Similar Laptops from Dataset")

similar = (
    (old_df['Company'] == company) &
    (old_df['TypeName'] == type) &
    (old_df['Ram'].str.contains(str(ram))) &
    (old_df['Cpu'].str.contains(str(cpu))) &
    (old_df['Gpu'].str.contains(str(gpu)))
)

st.dataframe(old_df[similar])
