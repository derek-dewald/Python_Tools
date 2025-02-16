import sys
import os
import pandas as pd
import streamlit as st

# Get path to the root folder (Python_Tools)
base_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))

# Add the `d_py_functions` folder to the system path
sys.path.append(os.path.join(base_dir, "d_py_functions"))

# Now import your functions
from Organization import CreateMarkdownfromProcess
from Connections import ParamterMapping

# List of processes
p_list = ParamterMapping('ProcessSheet')['Process'].unique().tolist()
p_list = [item for item in p_list if pd.notna(item)]

# Dropdown to select the process
selected_process = st.selectbox("Select a Process:", p_list)

if st.button("Generate Markdown"):
    markdown_text = CreateMarkdownfromProcess(selected_process,return_value='text')
    
    if markdown_text:
        st.markdown(markdown_text, unsafe_allow_html=True)
    else:
        st.error("No data found for the selected process.")