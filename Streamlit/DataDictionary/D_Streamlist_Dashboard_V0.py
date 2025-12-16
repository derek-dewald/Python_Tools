import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Python Function Catalog", layout="wide")

# ✅ Full-width container override
st.markdown(
    """
    <style>
      .block-container {
        max-width: 100% !important;
        padding-left: 1rem;
        padding-right: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Data sources (raw GitHub)
# -----------------------
function_list_url = (
    "https://raw.githubusercontent.com/"
    "derek-dewald/Python_Tools/main/Streamlit/DataDictionary/python_function_list.csv"
)
parameter_list_url = (
    "https://raw.githubusercontent.com/"
    "derek-dewald/Python_Tools/main/Streamlit/DataDictionary/python_function_parameters.csv"
)

@st.cache_data(show_spinner=False)
def load_data():
    function_list_df = pd.read_csv(function_list_url)
    parameter_list_df = pd.read_csv(parameter_list_url)

    # ✅ Folder first, then Function
    function_list_df1 = function_list_df[["Folder", "Function", "Purpose"]].copy()
    parameter_list_df1 = parameter_list_df[["Folder", "Function", "Parameters", "Definition"]].copy()

    # ensure text columns are strings
    for c in ["Folder", "Function", "Purpose"]:
        function_list_df1[c] = function_list_df1[c].fillna("").astype(str)

    for c in ["Folder", "Function", "Parameters", "Definition"]:
        parameter_list_df1[c] = parameter_list_df1[c].fillna("").astype(str)

    return function_list_df1, parameter_list_df1

function_list_df1, parameter_list_df1 = load_data()

# -----------------------
# Navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Function List", "Function Parameters"])

# -------------------------
# Function List
# -------------------------
if page == "Function List":
    st.title("🧰 Function List")
    df_base = function_list_df1.copy()

    st.subheader("🔽 Filters")
    c1, c2, c3 = st.columns([1, 1, 2])

    # 1) Folder slicer from full df
    with c1:
        folder_opts = ["(All)"] + sorted([x for x in df_base["Folder"].unique() if x.strip()])
        sel_folder = st.selectbox("Folder", folder_opts, index=0)

    # Apply folder
    df1 = df_base if sel_folder == "(All)" else df_base[df_base["Folder"] == sel_folder]

    # 2) Function slicer depends on folder selection
    with c2:
        func_opts = ["(All)"] + sorted([x for x in df1["Function"].unique() if x.strip()])
        sel_func = st.selectbox("Function", func_opts, index=0)

    # Apply function
    df2 = df1 if sel_func == "(All)" else df1[df1["Function"] == sel_func]

    # 3) Purpose search (Purpose only) – does not change slicer lists unless you want it to
    with c3:
        purpose_search = st.text_input(
            "Purpose search",
            value="",
            placeholder="Type to search Purpose..."
        )

    df_view = df2
    if purpose_search.strip():
        s = purpose_search.strip().lower()
        df_view = df_view[df_view["Purpose"].str.lower().str.contains(s, na=False)]

    st.caption(f"Rows: {len(df_view)}")

    # Grid
    gb = GridOptionsBuilder.from_dataframe(df_view)
    gb.configure_column("Folder", width=180)
    gb.configure_column("Function", width=220)
    gb.configure_column("Purpose", flex=1, wrapText=True, autoHeight=True)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)

    AgGrid(
        df_view,
        gridOptions=gb.build(),
        height=800,
        fit_columns_on_grid_load=True
    )

# -----------------------------------
# Function Parameters
# -----------------------------------
elif page == "Function Parameters":
    st.title("🧩 Function Parameters")
    df_base = parameter_list_df1.copy()

    st.subheader("🔽 Filters")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    # 1) Folder slicer from full df
    with c1:
        folder_opts = ["(All)"] + sorted([x for x in df_base["Folder"].unique() if x.strip()])
        sel_folder = st.selectbox("Folder", folder_opts, index=0)

    # Apply folder
    df1 = df_base if sel_folder == "(All)" else df_base[df_base["Folder"] == sel_folder]

    # 2) Function slicer depends on folder
    with c2:
        func_opts = ["(All)"] + sorted([x for x in df1["Function"].unique() if x.strip()])
        sel_func = st.selectbox("Function", func_opts, index=0)

    # Apply function
    df2 = df1 if sel_func == "(All)" else df1[df1["Function"] == sel_func]

    # 3) Parameters slicer depends on folder+function
    with c3:
        param_opts = ["(All)"] + sorted([x for x in df2["Parameters"].unique() if x.strip()])
        sel_param = st.selectbox("Parameters", param_opts, index=0)

    # Apply parameters
    df3 = df2 if sel_param == "(All)" else df2[df2["Parameters"] == sel_param]

    # 4) Definition search (Definition only)
    with c4:
        definition_search = st.text_input(
            "Definition search",
            value="",
            placeholder="Type to search Definition..."
        )

    df_view = df3
    if definition_search.strip():
        s = definition_search.strip().lower()
        df_view = df_view[df_view["Definition"].str.lower().str.contains(s, na=False)]

    st.caption(f"Rows: {len(df_view)}")

    # Grid
    gb = GridOptionsBuilder.from_dataframe(df_view)
    gb.configure_column("Folder", width=180)
    gb.configure_column("Function", width=220)
    gb.configure_column("Parameters", width=320)
    gb.configure_column("Definition", flex=1, wrapText=True, autoHeight=True)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)

    AgGrid(
        df_view,
        gridOptions=gb.build(),
        height=800,
        fit_columns_on_grid_load=True
    )
