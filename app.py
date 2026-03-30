import streamlit as st
import torch
import pandas as pd
import numpy as np
from gan_model import Generator
from streamlit_option_menu import option_menu

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="CGAN Fraud App", layout="wide")

# ======================
# SIDEBAR MENU
# ======================
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Generate Data", "Analytics", "About"],
        icons=["house", "cpu", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# ======================
# LOAD MODEL
# ======================
latent_dim = 100

@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model

G = load_model()

# ======================
# HOME PAGE
# ======================
if selected == "Home":
    st.title("💳 AI-based Fraud Detection using Synthetic Data (CGAN)")
    st.caption("Synthetic Data Generation for Fraud Detection")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "97%")
    col2.metric("Fraud Recall", "68%")
    col3.metric("F1 Score", "0.80")

    st.markdown("---")
    st.subheader("Project Overview")
    st.image("Figure_1.png")

    st.write("""
    This project uses Conditional GAN (CGAN) to generate synthetic financial transaction data.
    The generated data helps improve fraud detection models by addressing class imbalance.
    """)

# ======================
# GENERATE DATA PAGE
# ======================
elif selected == "Generate Data":
    st.title("🔹 Generate Synthetic Transaction")

    if st.button("Generate Data"):
        noise = torch.randn(1, latent_dim)
        label = torch.tensor([[1.0]])

        with st.spinner("Generating synthetic data..."):
            with torch.no_grad():
                fake = G(noise, label).numpy()

        df = pd.DataFrame(fake)

        st.success("Synthetic Data Generated Successfully!")
        st.dataframe(df)
        
# ======================
# ANALYTICS PAGE
# ======================
elif selected == "Analytics":
    st.title("📊 Data Visualization")

    st.subheader("Real vs Synthetic Distribution")

    try:
        st.image("Figure_1.png")
        st.image("Figure_2.png")
        st.image("Figure_3.png")
        st.image("Figure_4.png")
    except:
        st.warning("Graphs not available")

    st.markdown("---")

    st.subheader("Sample Real Data")

    try:
        df_real = pd.read_csv("processed_data.csv").sample(5)
        st.dataframe(df_real)
    except:
        st.warning("Real dataset not available in deployed version.")
        df_dummy = pd.DataFrame(np.random.rand(5, 10))
        st.dataframe(df_dummy)

# ======================
# ABOUT PAGE
# ======================
elif selected == "About":
    st.title("ℹ️ About Project")

    st.write("""
    This project implements a Conditional Generative Adversarial Network (CGAN) 
    for synthetic data generation in fraud detection systems.

    Key Components:
    - Data Preprocessing
    - GAN Training (Generator + Discriminator)
    - Synthetic Data Generation
    - Model Evaluation
    - Deployment using Streamlit

    Purpose:
    Improve fraud detection performance using synthetic data augmentation.
    """)

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("Developed by Surya Pratap Singh")