import streamlit as st
import pandas as pd


import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'd_py_functions')))

from Organization import CreateMarkdownfromProcess

process_list = [
    "Machine Learning Project",
    "BLUE"]

st.title("📋 Process Documentation Dashboard")

for process in process_list:
    try:
        html_text = CreateMarkdownfromProcess(process, return_value="text")
        if html_text:
            with st.expander(f"🔹 {process}"):
                st.markdown(html_text, unsafe_allow_html=True)
        else:
            st.warning(f"No content found for '{process}'")
    except Exception as e:
        st.error(f"Error displaying '{process}': {e}")