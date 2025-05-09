from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import ast
import sys

# Ensure your custom function directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'd_py_functions')))

from Organization import CreateMarkdownfromProcess

def show_documented_processes_reference():
    st.title("📘 D's Documented Processes (For Reference)")

    # Load process list
    try:
        df = pd.read_csv(
            'https://docs.google.com/spreadsheets/d/e/2PACX-1vSbgjQNDbwl_UjsXd-zN6dCDofE_mdHJli1kPPp5bmv6gagoT8CEGMa38UWdJ4B9GHXd_ULozunfX1h/pub?output=csv'
        )
        process_list = df['Process'].dropna().unique().tolist()
    except Exception as e:
        st.error(f"Failed to load process list: {e}")
        return

    # Show each process as expandable markdown section
    for process in process_list:
        try:
            html_text = CreateMarkdownfromProcess(process, return_value="text")
            if html_text:
                with st.expander(f"🔹 {process}"):
                    st.markdown(html_text, unsafe_allow_html=True)
            else:
                st.warning(f"No content found for '{process}'")
        except Exception as e:
            st.error(f"❌ Error displaying '{process}': {e}")

def show_keyword_reference_browser():
    st.title("🔍 Key Term Search and Reference")

    # Load data from Google Sheets
    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?gid=0&single=true&output=csv'
    df = pd.read_csv(url)

    # Text-based filter input
    st.subheader("Key Word / Phrase Search")
    search_query = st.text_input("Search:", placeholder="Type to search...")

    visible_columns = ['Word', 'Category', 'Sub Categorization']

    # Apply text filter
    if search_query:
        filtered_df = df[
            df.apply(
                lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(),
                axis=1,
            )
        ][visible_columns]
    else:
        filtered_df = df[visible_columns]

    filtered_df = filtered_df.reset_index(drop=False)

    # Configure AgGrid
    builder = GridOptionsBuilder.from_dataframe(filtered_df)
    builder.configure_default_column(
        wrapText=True,
        autoHeight=True,
        cellStyle={'textAlign': 'center'}
    )
    builder.configure_selection("single", use_checkbox=False)
    builder.configure_column("index", hide=True)
    grid_options = builder.build()

    st.subheader("📋 Key Terms and Classification")
    response = AgGrid(
        filtered_df,
        gridOptions=grid_options,
        height=300,
        width="100%",
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED
    )

    selected_rows = response.get("selected_rows", [])

    try:
        selected_data = pd.DataFrame(selected_rows)
        if not selected_data.empty:
            final_df = df.merge(selected_data[['Word']], on='Word', how='inner')
            transposed_df = pd.DataFrame({
                "Field": final_df.columns,
                "Value": final_df.iloc[0]
            }).fillna("")

            # Format links and cleanup
            transposed_df["Value"] = transposed_df.apply(
                lambda row: f"[Open Link]({row['Value']})" if row["Field"] == "Link" and row["Value"] else row['Value'],
                axis=1
            )
            transposed_df['Value'] = np.where(
                transposed_df['Value'] == '[Open Link](nan)', "", transposed_df['Value']
            )

            st.subheader("📑 Key Term Reference Details")
            for _, row in transposed_df.iterrows():
                field = row["Field"]
                value = row["Value"]

                if field == "Link":
                    st.markdown(f"**{field}:** {value}")
                elif field == "Image":
                    if value and not pd.isna(value):
                        st.image(value, caption="Image Reference", width=300)
                    else:
                        st.warning("⚠️ No image available.")
                elif field == "Markdown":
                    st.latex(value)
                else:
                    st.write(f"**{field}:**\n{value}")
        else:
            st.info("Select a row above to view details.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

    st.markdown("""
    ---
    🔗 [Open Raw Data in Google Sheets](https://docs.google.com/spreadsheets/d/1tZ-_5Vv99_bm9CCEdDDN0KkmsFNcjWeKM86237yrCTQ/edit?gid=0#gid=0)
    """)

# --- Configuration for each page ---
column_dict1 = {'Description':300,'Model':150,'Name':100}
column_order1 = ['Model','Name','Default','Type','Description','Section']

column_dict2 = {'Model Name':150,'Sklearn Desc':350}
column_order2 = ['Model Name','Estimator Type','Dataset Size','Full Class Path','Sklearn Desc']


GITHUB_USER = "derek-dewald"
GITHUB_REPO = "Python_Tools"
FOLDER_PATH = "d_py_functions"
LOCAL_DIR = "./downloaded_py_files/"

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

def show_function_browser():
    st.title("🧰 Python Function Explorer")

    with st.spinner("🔄 Loading Python files from GitHub..."):
        fetch_and_save_python_files()
        df = read_python_files()

    df_display = df.drop(columns=["Code", "Arguments", "Return"])
    df_display = df_display.loc[:, ~df_display.columns.duplicated()].copy()

    files = ["Show All"] + sorted(df["File"].unique().tolist())
    selected_file = st.selectbox("📁 Filter by File", files, index=0)

    if selected_file != "Show All":
        df_display = df_display[df_display["File"] == selected_file]

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

    st.markdown("### 🔍 Click a function to view its code")
    response = AgGrid(
        df_display,
        gridOptions=grid_options,
        height=400,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED
    )

    selected_rows = response.get("selected_rows")
    selected_function = None

    if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
        selected_function = selected_rows.iloc[0]["Function Name"]
    elif isinstance(selected_rows, list) and len(selected_rows) > 0:
        selected_function = selected_rows[0].get("Function Name")

    if selected_function:
        function_row = df[df["Function Name"] == selected_function]
        function_code = function_row["Code"].values[0]
        st.code(function_code, language="python")

def show_ds_coding_dashboard():
    # Load data from Google Sheets
    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSnwd-zccEOQbpNWdItUG0qXND5rPVFbowZINjugi15TdWgqiy3A8eMRhbmSMBiRhHt1Qsry3E8tKY8/pub?output=csv'
    df = pd.read_csv(url)

    # Clean Description column
    df['Description'] = df['Description'].fillna("").astype(str)

    st.title("📖 DS Coding Dashboard")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        programs = ["All"] + sorted(df['Program'].dropna().unique().tolist())
        selected_program = st.selectbox("Select Program", programs)
    with col2:
        classifications = ["All"] + sorted(df['Classification'].dropna().unique().tolist())
        selected_classification = st.selectbox("Select Classification", classifications)
    with col3:
        descriptions = ["All"] + sorted(df['Description'].dropna().unique().tolist())
        selected_description = st.selectbox("Select Description", descriptions)

    # Apply filters
    filtered_df = df.copy()
    if selected_program != "All":
        filtered_df = filtered_df[filtered_df['Program'] == selected_program]
    if selected_classification != "All":
        filtered_df = filtered_df[filtered_df['Classification'] == selected_classification]
    if selected_description != "All":
        filtered_df = filtered_df[filtered_df['Description'] == selected_description]

    # Drop unneeded columns for display
    display_df = filtered_df.drop(['Program', 'Classification'], axis=1)

    # Configure AgGrid
    builder = GridOptionsBuilder.from_dataframe(display_df)
    builder.configure_column('Command_Code', width=350, wrapText=True, suppressSizeToFit=True, cellStyle={'whiteSpace': 'normal', 'textAlign': 'center'})
    builder.configure_column('Description', width=200, wrapText=True, suppressSizeToFit=True, cellStyle={'whiteSpace': 'normal', 'textAlign': 'center'})
    builder.configure_column('Comments', width=800, wrapText=True, suppressSizeToFit=True, cellStyle={'whiteSpace': 'normal', 'textAlign': 'center'})

    builder.configure_default_column(
        wrapText=True,
        autoHeight=True,
        resizable=True,
        cellStyle={
            'textAlign': 'center',
            'whiteSpace': 'normal',
            'wordBreak': 'break-word'
        }
    )

    grid_options = builder.build()

    # Display
    st.subheader("Filtered Results")
    AgGrid(
        display_df,
        gridOptions=grid_options,
        fit_columns_on_grid_load=True,
        height=400,
        allow_unsafe_jscode=True
    )

    st.markdown(
        """
        ### Relevant Links
        [Raw Data](https://docs.google.com/spreadsheets/d/1FpYYq4LN6AZBaNRhnj1f76YNvnG-hTco40wJ1PUugto/edit?gid=0#gid=0)
        """
    )


def StreamlitBaseDashboard(google_drive_csv,
                            filter_columns,
                            title,
                            column_order,
                            column_widths=None,
                            default_width=100):

    st.title(title)

    df = pd.read_csv(google_drive_csv)
    df.columns = df.columns.str.strip()
    df = df[column_order]  # Ensure column order after stripping headers

    st.subheader("🔽 Filters")
    filtered_df = df.copy()
    for col in filter_columns:
        if col in df.columns:
            unique_vals = sorted(df[col].dropna().unique())
            selected_val = st.selectbox(f"Select {col}", ["(All)"] + unique_vals)
            if selected_val != "(All)":
                filtered_df = filtered_df[filtered_df[col] == selected_val]
        else:
            st.warning(f"Column '{col}' not found in the data.")

    st.subheader("📋 Filtered Table with Wrapped Text")
    gb = GridOptionsBuilder.from_dataframe(filtered_df)
    gb.configure_default_column(wrapText=True, autoHeight=True)

    applied_columns = set()
    if column_widths:
        for col, width in column_widths.items():
            if col in filtered_df.columns:
                gb.configure_column(col, width=width)
                applied_columns.add(col)

    for col in filtered_df.columns:
        if col not in applied_columns:
            gb.configure_column(col, width=default_width)

    grid_options = gb.build()
    AgGrid(filtered_df, gridOptions=grid_options, height=600, fit_columns_on_grid_load=False)


# ✅ Must be the first Streamlit command
st.set_page_config(page_title="SKLearn Dashboards", layout="wide")


# Sidebar navigation — this is now the only entry point
page = st.sidebar.selectbox("Select Page", ['Data Dictionary',"Code Dashboard","Python Function Dashboard",'Process Dashboard',"SKLearn Model Viewer", "SKLearn Parameter Dashboard"])

if page == "SKLearn Parameter Dashboard":
    StreamlitBaseDashboard(
        google_drive_csv='https://docs.google.com/spreadsheets/d/e/2PACX-1vSnzcSYTXm2jl9GXvrNaH3b3TPbNufGJwRTMeJ8Ckhse_r9CWGjlXWlUfyGTwcnoXqT7ZLLyBAk2rKO/pub?gid=175044227&single=true&output=csv',
        filter_columns=['Model', 'Name'],
        title='SKLearn Parameter Viewer',
        column_widths=column_dict1,
        column_order=column_order1
    )
elif page == "SKLearn Model Viewer":
    StreamlitBaseDashboard(
        google_drive_csv='https://docs.google.com/spreadsheets/d/e/2PACX-1vToNab_ADzmxRZkg4bJ5wOwLZcrdwNYrxwWBETdGlfGSoUpnyy799EpbYqDqnwyKs2bEyJHUu58SX8Q/pub?gid=1125524518&single=true&output=csv',
        filter_columns=['Model Name','Dataset Size','Estimator Type'],
        title='SKLearn Model Viewer',
        column_widths=column_dict2,
        column_order=column_order2
    )

elif page =="Code Dashboard":
    show_ds_coding_dashboard()

elif page =='Python Function Dashboard':
    show_function_browser()

elif page =='Data Dictionary':
    show_keyword_reference_browser()


elif page =='Process Dashboard':
    show_documented_processes_reference()