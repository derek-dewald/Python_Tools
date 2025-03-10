import sys
import os
import pandas as pd
import streamlit as st

# Debugging: Print system path
st.write("🔍 Current sys.path:")
st.write(sys.path)

# Get path to the root folder (Python_Tools)
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../"))

# Add `d_py_functions` to the system path
D_PY_FUNCTIONS_DIR = os.path.join(BASE_DIR, "d_py_functions")

if os.path.exists(D_PY_FUNCTIONS_DIR):  # Check if folder exists in Streamlit
    st.write(f"✅ Found d_py_functions at: {D_PY_FUNCTIONS_DIR}")
    sys.path.append(D_PY_FUNCTIONS_DIR)
else:
    st.error(f"❌ d_py_functions not found at {D_PY_FUNCTIONS_DIR}")

# Now try to import
try:
    from Organization import CreateMarkdownfromProcess
    from Connections import ParamterMapping
    st.write("✅ Successfully imported modules!")
except ImportError as e:
    st.error(f"❌ ImportError: {e}")

# Fetch available processes
try:
    p_list = ParamterMapping('ProcessSheet')['Process'].unique().tolist()
    p_list = [item for item in p_list if pd.notna(item)]
except Exception as e:
    st.error(f"Error fetching process list: {e}")
    p_list = []

# Streamlit UI
st.title("📄 Markdown Generator for Processes")

# Dropdown to select a process
selected_process = st.selectbox("Select a Process:", p_list)

if st.button("Generate Markdown"):
    try:
        markdown_text = CreateMarkdownfromProcess(selected_process, return_value='text')
        
        if markdown_text:
            st.markdown(markdown_text, unsafe_allow_html=True)
        else:
            st.error("No data found for the selected process.")
    except Exception as e:
        st.error(f"Error generating Markdown: {e}")