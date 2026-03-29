import streamlit as st
import torch
import pandas as pd
from gan_model import Generator

st.title("Fraud Detection using Synthetic Data (CGAN)")

latent_dim = 100

# Load Generator
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model

G = load_model()

# ======================
# Generate Synthetic Data
# ======================

st.subheader("Generate Synthetic Transaction")

if st.button("Generate Data"):
    noise = torch.randn(1, latent_dim)
    label = torch.tensor([[1.0]])

    with torch.no_grad():
        fake = G(noise, label).numpy()

    df = pd.DataFrame(fake)

    st.write("Synthetic Transaction:")
    st.dataframe(df)

# ======================
# Show Dataset Sample
# ======================

st.subheader("Sample Real Data")

df_real = pd.read_csv("processed_data.csv").sample(5)
st.dataframe(df_real)