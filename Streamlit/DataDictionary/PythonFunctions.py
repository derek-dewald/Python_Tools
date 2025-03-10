import streamlit as st
import requests
import os
import pandas as pd
import ast
from st_aggrid import AgGrid, GridOptionsBuilder

# Define GitHub repo details
GITHUB_USER = "derek-dewald"
GITHUB_REPO = "Python_Tools"
FOLDER_PATH = "d_py_functions"
LOCAL_DIR = "./downloaded_py_files/"

# Ensure directory exists
os.makedirs(LOCAL_DIR, exist_ok=True)

@st.cache_data
def fetch_and_save_python_files():
    """Fetch Python files from GitHub and save them locally."""
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
    """Extract structured function details from Python script using AST."""
    tree = ast.parse(file_content)
    function_data = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            docstring = ast.get_docstring(node) or "No description available"
            args = [arg.arg for arg in node.args.args]
            return_type = ast.unparse(node.returns) if node.returns else "None"
            function_code = ast.get_source_segment(file_content, node).strip()

            description_text = []
            args_text = []
            return_text = "None"
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
                "Function Name": function_name,  # ✅ Fix: Ensure this column exists only once
                "Description": "\n".join(description_text).strip(),
                "Arguments": ", ".join(args) if args else "None",
                "Return": return_text,
                "Code": function_code,
                "File": file_name
            }
            
    return function_data

@st.cache_data
def read_python_files(location=LOCAL_DIR):
    """Reads all Python files and extracts function details."""
    py_file_dict = {}
    for file_name in os.listdir(location):
        if file_name.endswith(".py") and "__" not in file_name:
            with open(os.path.join(location, file_name), "r", encoding="utf-8") as file:
                data = file.read()
                py_file_dict.update(extract_function_details_ast(data, file_name))
    
    df = pd.DataFrame(py_file_dict).T  # Transpose to match expected format

    # ✅ Fix: Ensure 'Function Name' is correctly set
    if "Function Name" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Function Name"}, inplace=True)

    return df

# Streamlit App UI
st.set_page_config(layout="wide")
# Automatically fetch and process files when the app loads
with st.spinner("🔄 Loading Python files..."):
    fetch_and_save_python_files()  # Fetch files on load
    df = read_python_files()  # Load dataframe

# Drop unnecessary columns to save space in AgGrid display
df_display = df.drop(columns=["Code", "Arguments",'Return'])

# **File Filter Dropdown**
files = ["Show All"] + sorted(df["File"].unique().tolist())  # "Show All" as default
selected_file = st.selectbox("📁 Filter by File:", files, index=0)

# Apply file filter if a specific file is selected
if selected_file != "Show All":
    df_display = df_display[df_display["File"] == selected_file]

# **Fix: Ensure columns are unique before passing to AgGrid**
df_display = df_display.loc[:, ~df_display.columns.duplicated()].copy()

# **Configure AgGrid Options**
builder = GridOptionsBuilder.from_dataframe(df_display)
# **Set Specific Column Widths**
builder.configure_column("Function Name", width=120)
builder.configure_column("Description", width=800)
builder.configure_column("File", width=120)
builder.configure_selection("single")  # Allow single row selection
builder.configure_default_column(
    wrapText=True,  # Enable text wrapping
    autoHeight=True,  # Adjust row height automatically
    cellStyle={'textAlign': 'left'}
)
grid_options = builder.build()

# **Show AgGrid Table**
st.markdown("### 🔍 Click a Function to View Details")
response = AgGrid(
    df_display,
    gridOptions=grid_options,
    height=400,
    width="100%",
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True
)

# **Get Selected Row**
selected_rows = response.get("selected_rows", [])

try:
    selected_function = selected_rows['Function Name'].item()
    function_row = df[df["Function Name"] == selected_function]
    function_code = function_row["Code"].values[0]  # ✅ Extract Code column correctly
    st.code(function_code, language="python")  # ✅ Display code block
                      
except:
    pass
