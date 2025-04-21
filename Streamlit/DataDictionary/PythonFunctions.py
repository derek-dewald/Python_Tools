import streamlit as st
import requests
import os
import pandas as pd
import ast
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# GitHub Repo Details
GITHUB_USER = "derek-dewald"
GITHUB_REPO = "Python_Tools"
FOLDER_PATH = "d_py_functions"
LOCAL_DIR = "./downloaded_py_files/"

# Ensure directory exists
os.makedirs(LOCAL_DIR, exist_ok=True)

@st.cache_data
def fetch_and_save_python_files():
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{FOLDER_PATH}"
    response = requests.get(api_url)

    if response.status_code == 200:
        files = response.json()
        py_files = [file for file in files if file['name'].endswith('.py')]

        for file in py_files:
            file_url = file['download_url']
            file_name = file['name']
            file_response = requests.get(file_url)
            if file_response.status_code == 200:
                with open(os.path.join(LOCAL_DIR, file_name), "w", encoding="utf-8") as f:
                    f.write(file_response.text)
        return True
    else:
        return False

def extract_function_details_ast(file_content, file_name):
    tree = ast.parse(file_content)
    function_data = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            docstring = ast.get_docstring(node) or "No description available"
            args = [arg.arg for arg in node.args.args]
            return_type = ast.unparse(node.returns) if node.returns else "None"
            function_code = ast.get_source_segment(file_content, node).strip()

            description_text, args_text, return_text = [], [], "None"
            if docstring:
                doc_lines = docstring.split("\n")
                found_args = False
                for line in doc_lines:
                    stripped = line.strip()
                    if stripped.lower().startswith(("args:", "parameters:")):
                        found_args = True
                        continue
                    elif stripped.lower().startswith("returns:"):
                        return_text = stripped.replace("Returns:", "").strip()
                        found_args = False
                        continue
                    if not found_args:
                        description_text.append(stripped)
                    else:
                        args_text.append(stripped)

            function_data[function_name] = {
                "Function Name": function_name,
                "Description": "\n".join(description_text).strip(),
                "Arguments": ", ".join(args) if args else "None",
                "Return": return_text,
                "Code": function_code,
                "File": file_name
            }

    return function_data

@st.cache_data
def read_python_files(location=LOCAL_DIR):
    py_file_dict = {}
    for file_name in os.listdir(location):
        if file_name.endswith(".py") and "__" not in file_name:
            with open(os.path.join(location, file_name), "r", encoding="utf-8") as file:
                data = file.read()
                py_file_dict.update(extract_function_details_ast(data, file_name))

    df = pd.DataFrame(py_file_dict).T

    if "Function Name" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Function Name"}, inplace=True)

    return df

# Streamlit App
st.set_page_config(layout="wide")

with st.spinner("🔄 Loading Python files..."):
    fetch_and_save_python_files()
    df = read_python_files()

df_display = df.drop(columns=["Code", "Arguments", "Return"])
files = ["Show All"] + sorted(df["File"].unique().tolist())
selected_file = st.selectbox("📁 Filter by File:", files, index=0)

if selected_file != "Show All":
    df_display = df_display[df_display["File"] == selected_file]

df_display = df_display.loc[:, ~df_display.columns.duplicated()].copy()

# GridOptionsBuilder config
builder = GridOptionsBuilder.from_dataframe(df_display)
builder.configure_column("Function Name", width=120)
builder.configure_column("Description", width=800)
builder.configure_column("File", width=120)
builder.configure_selection("single")
builder.configure_default_column(
    wrapText=True,
    autoHeight=True,
    cellStyle={'textAlign': 'left'}
)
grid_options = builder.build()

# Render AgGrid
st.markdown("### 🔍 Click a Function to View Details")
response = AgGrid(
    df_display,
    gridOptions=grid_options,
    height=400,
    width="100%",
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    update_mode=GridUpdateMode.SELECTION_CHANGED  # ✅ Required!
)

selected_rows = response.get("selected_rows")

if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
    selected_function = selected_rows.iloc[0]["Function Name"]
elif isinstance(selected_rows, list) and len(selected_rows) > 0:
    selected_function = selected_rows[0].get("Function Name")
else:
    selected_function = None

if selected_function:
    function_row = df[df["Function Name"] == selected_function]
    function_code = function_row["Code"].values[0]
    st.code(function_code, language="python")
