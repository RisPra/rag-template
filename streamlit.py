import streamlit as st

st.title("My RAG Model")

models = (
    "gemini-1.0-pro",
    "OrcaMaidXL-17B-32k-GPTQ"
)

st.selectbox("Pick a model: ", models)