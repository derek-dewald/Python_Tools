import streamlit as st
import pandas as pd
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'd_py_functions')))

from Organization import CreateMarkdownfromProcess

df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSbgjQNDbwl_UjsXd-zN6dCDofE_mdHJli1kPPp5bmv6gagoT8CEGMa38UWdJ4B9GHXd_ULozunfX1h/pub?output=csv')

process_list = df['Process'].unique().tolist()

st.title("D's Documented Processes (For Reference)")

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