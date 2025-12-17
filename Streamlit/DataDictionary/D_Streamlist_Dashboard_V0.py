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




@st.cache_data(show_spinner=False)
def load_data():
    # Link to Function Listing, Manually Updated.
    function_list_url = (
        "https://raw.githubusercontent.com/"
        "derek-dewald/Python_Tools/main/Streamlit/DataDictionary/python_function_list.csv"
    )

    # Link To Function Parameter List, which is Manually Updated. 
    parameter_list_url = (
        "https://raw.githubusercontent.com/"
        "derek-dewald/Python_Tools/main/Streamlit/DataDictionary/python_function_parameters.csv"
    )

    google_note_csv = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSQF2lNc4WPeTRQ_VzWPkqSZp4RODFkbap8AqmolWp5bKoMaslP2oRVVG21x2POu_JcbF1tGRcBgodu/pub?output=csv'
    google_definition_csv = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv'
    
    data_dict = {}
    
    data_dict['google_notes_df'] = pd.read_csv(google_note_csv)
    data_dict['google_definition_df'] = pd.read_csv(google_definition_csv)
    data_dict['function_list_df'] = pd.read_csv(function_list_url)
    data_dict['parameter_list_df'] = pd.read_csv(parameter_list_url)

    # ✅ Folder first, then Function
    data_dict['function_list_df1']  = data_dict['function_list_df'][["Folder", "Function", "Purpose"]].copy()
    data_dict['parameter_list_df1'] = data_dict['parameter_list_df'][["Folder", "Function", "Parameters", "Definition"]].copy()

    for dict_key in data_dict.keys():
        for column in data_dict[dict_key].columns:
            data_dict[dict_key][column] = data_dict[dict_key][column].fillna("").astype(str)

    return data_dict

data_dict = load_data()

# -----------------------
# Navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Function List", "Function Parameters",'D Notes','D Definitions'])

# -------------------------
# Function List
# -------------------------
if page == "Function List":
    st.title("Function List")
    df_base = data_dict['function_list_df1'].copy()
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
    st.title("Function Parameters")
    df_base = data_dict['parameter_list_df1'].copy()
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

# -----------------------------------
# D Notes
# -----------------------------------
elif page == 'D Notes':
    st.title("D Notes")
    df_base = data_dict['google_notes_df'].copy()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    # 1) Folder slicer from full df

    c1_word = 'Category'
    c2_word = 'Categorization'
    c3_word = 'Word'
    search_word = 'Description'

    with c1:
        c1_options = ["(All)"] + sorted([x for x in df_base[c1_word].unique() if x.strip()])
        c1_sel = st.selectbox(c1_word, c1_options, index=0)

    # Apply folder
    df1 = df_base if c1_sel == "(All)" else df_base[df_base[c1_word] == c1_sel]

    # 2) Function slicer depends on folder
    with c2:
        c2_options = ["(All)"] + sorted([x for x in df1[c2_word].unique() if x.strip()])
        c2_sel = st.selectbox(c2_word, c2_options, index=0)

    # Apply function
    df2 = df1 if c2_sel == "(All)" else df1[df1[c2_word] == c2_sel]

    # 3) Parameters slicer depends on folder+function
    with c3:
        c3_options = ["(All)"] + sorted([x for x in df2[c3_word].unique() if x.strip()])
        c3_sel = st.selectbox(c3_word, c3_options, index=0)

    # Apply function
    df3 = df2 if c3_sel == "(All)" else df2[df2[c3_word] == c3_sel]

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
        df_view = df_view[df_view[search_word].str.lower().str.contains(s, na=False)]

    st.caption(f"Rows: {len(df_view)}")

    # Grid
    gb = GridOptionsBuilder.from_dataframe(df_view)
    gb.configure_column(c1_word, width=180)
    gb.configure_column(c2_word, width=220)
    gb.configure_column(c3_word, width=320)
    gb.configure_column(search_word, flex=1, wrapText=True, autoHeight=True)
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

elif page == "D Definitions":
    st.title("D Definitions")
    df_base = data_dict['google_definition_df'].copy()
    st.write(df_base)