import streamlit as st

st.title("STS Dashboard Test")
st.write("If you can see this, Streamlit is working!")

if st.button("Test Button"):
    st.success("Button works!")

st.write("Python version:", __import__('sys').version)
st.write("Streamlit version:", st.__version__)
