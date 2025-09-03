# ---- Streamlit MUST be configured first ----
import streamlit as st
st.set_page_config(page_title="SKLearn Dashboards", layout="wide")

# ---- Imports ----
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd
import numpy as np
import requests
import os
import ast
import sys

# ---- Sidebar: quick cache + debug panel ----
with st.sidebar.expander("🧹 Cache / Debug"):
    if st.button("Clear data cache"):
        st.cache_data.clear()
        st.success("Cache cleared — rerun the app.")
    try:
        import platform
        st.caption("Versions")
        st.write({
            "python": platform.python_version(),
            "streamlit": st.__version__,
            "pandas": pd.__version__,
            "requests": requests.__version__,
        })
    except Exception:
        pass

# ---- Utilities ----
def read_csv_clean(url, required_cols=None):
    """Read CSV from URL, strip headers, and optionally enforce required columns."""
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing expected columns: {missing}")
            st.caption("Here are the columns we actually got:")
            st.write(list(df.columns))
            st.stop()
    return df

# Ensure your custom function directory is in the path (fail-soft import)
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'd_py_functions')))
except Exception:
    pass

try:
    from Organization import CreateMarkdownfromProcess
    _org_import_err = None
except Exception as e:
    CreateMarkdownfromProcess = None
    _org_import_err = str(e)

# ---- Tabs / Pages ----
def show_documented_processes_reference():
    st.title("📘 D's Documented Processes (For Reference)")

    if CreateMarkdownfromProcess is None:
        st.error(
            "Could not import `CreateMarkdownfromProcess` from Organization.py.\n\n"
            f"Details: {_org_import_err}"
        )
        return

    try:
        df = pd.read_csv(
            'https://docs.google.com/spreadsheets/d/e/2PACX-1vSbgjQNDbwl_UjsXd-zN6dCDofE_mdHJli1kPPp5bmv6gagoT8CEGMa38UWdJ4B9GHXd_ULozunfX1h/pub?output=csv'
        )
        df.columns = df.columns.str.strip()
        if 'Process' not in df.columns:
            st.error("Google Sheet is missing the 'Process' column.")
            st.write("Columns present:", list(df.columns))
            return
        process_list = df['Process'].dropna().unique().tolist()
    except Exception as e:
        st.error(f"Failed to load process list: {e}")
        return

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

    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?gid=0&single=true&output=csv'
    df = read_csv_clean(url, ['Word', 'Category', 'Sub Categorization'])

    st.subheader("Key Word / Phrase Search")
    search_query = st.text_input("Search:", placeholder="Type to search...")

    visible_columns = ['Word', 'Category', 'Sub Categorization']

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
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED
    )

    selected_rows = response.get("selected_rows", [])
    try:
        selected_data = pd.DataFrame(selected_rows)
        if not selected_data.empty:
            if 'Word' not in selected_data.columns:
                st.warning("Selection payload missing 'Word' column. Raw payload:")
                st.write(selected_data)
                return

            final_df = df.merge(selected_data[['Word']], on='Word', how='inner')
            # Safe transpose
            row0 = final_df.iloc[0].copy()
            transposed_df = row0.reset_index()
            transposed_df.columns = ["Field", "Value"]
            transposed_df = transposed_df.fillna("")

            # Format links and cleanup
            transposed_df["Value"] = transposed_df.apply(
                lambda r: f"[Open Link]({r['Value']})" if r["Field"] == "Link" and isinstance(r["Value"], str) and r["Value"].strip() else r['Value'],
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
                    if isinstance(value, str) and value.strip():
                        st.image(value, caption="Image Reference", width=300)
                    else:
                        st.warning("⚠️ No image available.")
                elif field == "Markdown":
                    st.markdown(value or "")
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


# --- Configuration for SKLearn dashboards ---
column_dict1 = {'Description': 300, 'Model': 150, 'Name': 100}
column_order1 = ['Model', 'Name', 'Default', 'Type', 'Description', 'Section']

column_dict2 = {'Model Name': 150, 'Sklearn Desc': 350}
column_order2 = ['Model Name', 'Estimator Type', 'Dataset Size', 'Full Class Path', 'Sklearn Desc']


GITHUB_USER = "derek-dewald"
GITHUB_REPO = "Python_Tools"
FOLDER_PATH = "d_py_functions"
LOCAL_DIR = "./downloaded_py_files/"

os.makedirs(LOCAL_DIR, exist_ok=True)

@st.cache_data
def fetch_and_save_python_files():
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{FOLDER_PATH}"
    response = requests.get(api_url, timeout=30)

    if response.status_code == 200:
        files = response.json() or []
        py_files = [file for file in files if isinstance(file, dict) and file.get('name', '').endswith('.py')]

        for file in py_files:
            try:
                file_url = file['download_url']
                file_name = file['name']
                file_response = requests.get(file_url, timeout=30)
                file_response.raise_for_status()
                with open(os.path.join(LOCAL_DIR, file_name), "w", encoding="utf-8") as f:
                    f.write(file_response.text)
            except Exception as e:
                st.warning(f"Skipped {file.get('name')}: {e}")
        return True
    else:
        st.error(f"GitHub API error {response.status_code}: {response.text[:200]}")
        return False

def extract_function_details_ast(file_content, file_name):
    tree = ast.parse(file_content)
    function_data = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            docstring = ast.get_docstring(node) or "No description available"
            args = [arg.arg for arg in node.args.args]
            return_type = ast.unparse(node.returns) if getattr(node, "returns", None) else "None"
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

    df_display = df.drop(columns=[c for c in ["Code", "Arguments", "Return"] if c in df.columns])
    df_display = df_display.loc[:, ~df_display.columns.duplicated()].copy()

    files = ["Show All"] + sorted(df["File"].dropna().astype(str).unique().tolist()) if "File" in df.columns else ["Show All"]
    selected_file = st.selectbox("📁 Filter by File", files, index=0)

    if selected_file != "Show All" and "File" in df_display.columns:
        df_display = df_display[df_display["File"] == selected_file]

    builder = GridOptionsBuilder.from_dataframe(df_display)
    if "Function Name" in df_display.columns:
        builder.configure_column("Function Name", width=160)
    if "Description" in df_display.columns:
        builder.configure_column("Description", width=800)
    if "File" in df_display.columns:
        builder.configure_column("File", width=160)
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
        selected_function = selected_rows.iloc[0].get("Function Name")
    elif isinstance(selected_rows, list) and len(selected_rows) > 0:
        selected_function = selected_rows[0].get("Function Name")

    if selected_function and "Function Name" in df.columns and "Code" in df.columns:
        function_row = df[df["Function Name"] == selected_function]
        if not function_row.empty:
            function_code = function_row["Code"].values[0]
            st.code(function_code, language="python")


def show_ds_coding_dashboard():
    url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSnwd-zccEOQbpNWdItUG0qXND5rPVFbowZINjugi15TdWgqiy3A8eMRhbmSMBiRhHt1Qsry3E8tKY8/pub?output=csv'
    df = read_csv_clean(url)

    # Ensure columns exist even if the sheet schema changed
    for c in ["Program", "Classification", "Description", "Command_Code", "Comments"]:
        if c not in df.columns:
            df[c] = ""
    df["Description"] = df["Description"].fillna("").astype(str)

    st.title("📖 DS Coding Dashboard")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        programs = ["All"] + sorted(df['Program'].dropna().astype(str).unique().tolist())
        selected_program = st.selectbox("Select Program", programs)
    with col2:
        classifications = ["All"] + sorted(df['Classification'].dropna().astype(str).unique().tolist())
        selected_classification = st.selectbox("Select Classification", classifications)
    with col3:
        descriptions = ["All"] + sorted(df['Description'].dropna().astype(str).unique().tolist())
        selected_description = st.selectbox("Select Description", descriptions)

    # Apply filters
    filtered_df = df.copy()
    if selected_program != "All":
        filtered_df = filtered_df[filtered_df['Program'] == selected_program]
    if selected_classification != "All":
        filtered_df = filtered_df[filtered_df['Classification'] == selected_classification]
    if selected_description != "All":
        filtered_df = filtered_df[filtered_df['Description'] == selected_description]

    # Drop unneeded columns for display (avoid KeyError)
    display_df = filtered_df.drop(['Program', 'Classification'], axis=1, errors="ignore")

    # Configure AgGrid
    builder = GridOptionsBuilder.from_dataframe(display_df)
    if "Command_Code" in display_df.columns:
        builder.configure_column('Command_Code', width=350, wrapText=True, suppressSizeToFit=True,
                                 cellStyle={'whiteSpace': 'normal', 'textAlign': 'center'})
    if "Description" in display_df.columns:
        builder.configure_column('Description', width=200, wrapText=True, suppressSizeToFit=True,
                                 cellStyle={'whiteSpace': 'normal', 'textAlign': 'center'})
    if "Comments" in display_df.columns:
        builder.configure_column('Comments', width=800, wrapText=True, suppressSizeToFit=True,
                                 cellStyle={'whiteSpace': 'normal', 'textAlign': 'center'})

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


def show_function_catalog_viewer():
    import io, re
    st.title("📒 Function Catalog Viewer")

    GITHUB_BLOB_URL = (
        "https://github.com/derek-dewald/Python_Tools/blob/main/"
        "d_py_functions/D_Python_Functions.xlsx"
    )

    def to_raw_github_url(url: str) -> str:
        m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
        if not m:
            return url
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"

    RAW_URL = to_raw_github_url(GITHUB_BLOB_URL)

    @st.cache_data(show_spinner=False)
    def fetch_workbook_bytes(raw_url: str) -> bytes:
        resp = requests.get(raw_url, timeout=30)
        resp.raise_for_status()
        return resp.content

    @st.cache_data(show_spinner=False)
    def read_first_sheet(xls_bytes: bytes) -> pd.DataFrame:
        with pd.ExcelFile(io.BytesIO(xls_bytes)) as xf:
            first = xf.sheet_names[0]
            df = xf.parse(sheet_name=first)
        for c in ["Date Created", "Date Last Modified"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    try:
        xls_bytes = fetch_workbook_bytes(RAW_URL)
        df = read_first_sheet(xls_bytes)
    except Exception as e:
        st.error(f"Failed to load the workbook: {e}")
        return

    preferred = [
        "File","Function Name","Description","Parameters","Returns","Raises",
        "Examples","Date Created","Date Last Modified","Return Type (Annotation)","Start Line","Code"
    ]
    df = df[[c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]]
    if "Code" in df.columns:
        df = df.drop(columns=["Code"])

    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    def build_date_options(series: pd.Series) -> list[str]:
        if series.dtype.kind == "M":
            vals = series.dropna().dt.date.astype(str).unique().tolist()
        else:
            s = pd.to_datetime(series, errors="coerce")
            vals = s.dropna().dt.date.astype(str).unique().tolist()
        return ["All","Blank"] + sorted(vals)

    with col1:
        if "File" in df.columns:
            file_opts = ["All"] + sorted(df["File"].dropna().astype(str).unique().tolist())
            sel_file = st.selectbox("File", file_opts, index=0)
        else:
            sel_file = "All"

    with col2:
        if "Date Created" in df.columns:
            dc_opts = build_date_options(df["Date Created"])
            sel_dc = st.selectbox("Date Created", dc_opts, index=0)
        else:
            sel_dc = "All"

    with col3:
        if "Date Last Modified" in df.columns:
            dlm_opts = build_date_options(df["Date Last Modified"])
            sel_dlm = st.selectbox("Date Last Modified", dlm_opts, index=0)
        else:
            sel_dlm = "All"

    view = df.copy()

    if sel_file != "All" and "File" in view.columns:
        view = view[view["File"].astype(str) == sel_file]

    def apply_date_filter(frame: pd.DataFrame, col: str, choice: str) -> pd.DataFrame:
        if col not in frame.columns or choice == "All":
            return frame
        if choice == "Blank":
            return frame[frame[col].isna() | (frame[col].astype(str).str.strip() == "")]
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            return frame[frame[col].dt.date.astype(str) == choice]
        col_parsed = pd.to_datetime(frame[col], errors="coerce")
        return frame[col_parsed.dt.date.astype(str) == choice]

    if sel_dc != "All":
        view = apply_date_filter(view, "Date Created", sel_dc)
    if sel_dlm != "All":
        view = apply_date_filter(view, "Date Last Modified", sel_dlm)

    search_text = st.text_input("Search text", "")
    if search_text:
        import re as _re
        pattern = _re.escape(search_text)
        searchable_cols = [c for c in ["Function Name","Description","Parameters","Returns","Raises","Examples","File"] if c in view.columns]
        if searchable_cols:
            mask = pd.Series(False, index=view.index)
            for c in searchable_cols:
                mask |= view[c].astype(str).str.contains(pattern, case=False, na=False)
            view = view[mask]

    st.divider()

    def wrap_df_for_display(df_in: pd.DataFrame) -> str:
        wrap_candidates = [c for c in ["Description","Parameters","Returns","Raises","Examples"] if c in df_in.columns]
        styles = []
        for col in wrap_candidates:
            styles.append(dict(selector=f"th.col_heading.level0#{col}", props=[("white-space","normal")]))
            styles.append(dict(selector=f"td.col#{col}", props=[("white-space","pre-wrap"),("word-break","break-word")]))
        styles.append(dict(selector="table", props=[("table-layout","fixed"),("width","100%")]))
        styles.append(dict(selector="tbody, thead", props=[("font-size","0.9rem")]))
        return (df_in.style.hide(axis="index").set_table_styles(styles, overwrite=False).set_properties(**{"white-space":"pre-wrap"})).to_html()

    if view.empty:
        st.info("No rows match the selected filters.")
    else:
        st.markdown(wrap_df_for_display(view), unsafe_allow_html=True)


def StreamlitBaseDashboard(google_drive_csv,
                           filter_columns,
                           title,
                           column_order,
                           column_widths=None,
                           default_width=100):

    st.title(title)

    df = read_csv_clean(google_drive_csv)
    df.columns = df.columns.str.strip()

    # Only keep present columns; warn if some are missing
    missing = [c for c in column_order if c not in df.columns]
    if missing:
        st.warning(f"Some expected columns were not found and will be skipped: {missing}")
    present_order = [c for c in column_order if c in df.columns]
    if not present_order:
        st.error("None of the expected columns are present. Cannot render table.")
        st.write("Columns present:", list(df.columns))
        return
    df = df[present_order]

    st.subheader("🔽 Filters")
    filtered_df = df.copy()
    for col in filter_columns:
        if col in filtered_df.columns:
            unique_vals = sorted(filtered_df[col].dropna().astype(str).unique())
            selected_val = st.selectbox(f"Select {col}", ["(All)"] + unique_vals)
            if selected_val != "(All)":
                filtered_df = filtered_df[filtered_df[col].astype(str) == selected_val]
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

# ---- Router (wrapped so errors appear in the UI) ----
page = st.sidebar.selectbox(
    "Select Page",
    [
        'Data Dictionary',
        "Code Dashboard",
        "Python Function Dashboard",
        'Process Dashboard',
        "SKLearn Model Viewer",
        "SKLearn Parameter Dashboard",
        'Raw Function Extract'
    ]
)

try:
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
            filter_columns=['Model Name', 'Dataset Size', 'Estimator Type'],
            title='SKLearn Model Viewer',
            column_widths=column_dict2,
            column_order=column_order2
        )
    elif page == "Code Dashboard":
        show_ds_coding_dashboard()
    elif page == 'Python Function Dashboard':
        show_function_browser()
    elif page == 'Data Dictionary':
        show_keyword_reference_browser()
    elif page == 'Process Dashboard':
        show_documented_processes_reference()
    elif page == "Raw Function Extract":
        show_function_catalog_viewer()
except Exception as e:
    st.exception(e)  # surface full traceback in-app
