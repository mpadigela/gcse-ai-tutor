import streamlit as st
from google import genai

# --- DISCREET CONFIGURATION ---
# Sets the browser tab name to "mplogic"
st.set_page_config(page_title="mplogic", layout="centered")

# Hide Streamlit UI elements (Menu, Footer, Header) for maximum stealth
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stAppDeployButton {display:none;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# --- LOAD SECRETS ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Config Error")
    st.stop()

# --- MINIMALIST INTERFACE ---
client = genai.Client(api_key=api_key)

# Providing a label for the system but collapsing it so it's invisible to the user
user_prompt = st.text_area(
    label="Input", 
    label_visibility="collapsed",
    placeholder="",
    height=300
)

# The nondescript "OK" button
if st.button("OK"):
# ... rest of your code remains the same ...
    if user_prompt:
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=user_prompt
            )
            
            st.divider()
            st.markdown(response.text)
            
        except Exception:
            # Generic error to avoid technical exposure
            st.error("Processing error.")