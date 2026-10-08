from __future__ import annotations

from st_aggrid import AgGrid, GridOptionsBuilder,GridUpdateMode, DataReturnMode,JsCode
import streamlit.components.v1 as components
import streamlit as st

import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
import textwrap
import html
import hashlib

# To Download Project Checklist 
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
from typing import Iterable, Optional
from io import BytesIO


# # Add to use Java to Auto adjust size of visuals.
on_grid_ready = JsCode("""
function(params) {
    setTimeout(function() {
        params.api.sizeColumnsToFit();
    }, 100);
}
""")

on_grid_size_changed = JsCode("""
function(params) {
    setTimeout(function() {
        params.api.sizeColumnsToFit();
    }, 100);
}
""")


def df_to_excel_bytes(df,
                      sheet_name= "Sheet1",
                      default_max_width= 30,
                      long_columns=[],
                      long_max_width= 80):
    """
    
    
    """
    min_width = 10
    padding = 2
    wrap_vertical_align= "top"
    freeze_header= True,

    long_columns_set = set(long_columns or [])

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        safe_sheet = sheet_name[:31]  # Excel sheet name limit
        df.to_excel(writer, index=False, sheet_name=safe_sheet)
        ws = writer.sheets[safe_sheet]

        if freeze_header:
            ws.freeze_panes = "A2"

        # Predefine alignments (reuse objects)
        wrap_align = Alignment(wrap_text=True, vertical=wrap_vertical_align)
        no_wrap_align = Alignment(wrap_text=False, vertical=wrap_vertical_align)

        for i, col in enumerate(df.columns, start=1):
            col_letter = get_column_letter(i)

            # Compute max string length in this column (including header)
            ser = df[col].astype(str).fillna("")
            max_len = max(len(str(col)), int(ser.map(len).max()) if len(ser) else 0)

            # Choose cap
            cap = long_max_width if col in long_columns_set else default_max_width

            # Proposed width (with padding)
            proposed = max_len + padding

            # Final width with min + cap
            final_width = max(min_width, min(proposed, cap))
            ws.column_dimensions[col_letter].width = final_width

            # Wrap if we had to cap (meaning content would exceed the allowed width)
            should_wrap = proposed > cap
            if should_wrap:
                # Apply wrap to entire column, incl header
                for cell in ws[col_letter]:
                    cell.alignment = wrap_align
            else:
                # Optional: set vertical alignment consistently
                for cell in ws[col_letter]:
                    cell.alignment = no_wrap_align

    return buffer.getvalue()


def display_reference_grid(
    table_df: pd.DataFrame,
    key: str,
    column_widths: dict[str, int],
    height: int = 350
) -> None:
    """
    Display an AgGrid reference table that refreshes when its data changes.
    """
    grid_df = table_df.copy().reset_index(drop=True)

    missing_columns = [
        column
        for column in column_widths
        if column not in grid_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns not found in DataFrame: {missing_columns}"
        )

    reference_options = GridOptionsBuilder.from_dataframe(grid_df)

    reference_options.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=True,
        autoHeight=True
    )

    for column, width in column_widths.items():
        reference_options.configure_column(
            column,
            width=width,
            minWidth=max(50, int(width * 0.75)),
            maxWidth=int(width * 1.5),
            wrapText=True,
            autoHeight=True
        )

    # Create a hash that changes whenever the DataFrame changes.
    dataframe_hash = hashlib.md5(
        pd.util.hash_pandas_object(
            grid_df,
            index=True
        ).values.tobytes()
    ).hexdigest()

    dynamic_key = f"{key}_{dataframe_hash}"

    AgGrid(
        grid_df,
        gridOptions=reference_options.build(),
        height=height,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        reload_data=True,
        key=dynamic_key
    )

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
        padding-top: 0.75rem;
        padding-bottom: 0.5rem;
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

    data_dict = {}

    technical_notes = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSnwd-zccEOQbpNWdItUG0qXND5rPVFbowZINjugi15TdWgqiy3A8eMRhbmSMBiRhHt1Qsry3E8tKY8/pub?output=csv'
    data_dict['technical_notes_df'] = pd.read_csv(technical_notes)

    try:
        knowledge_base_xlsx   = '/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/knowledge_base.xlsx' 
        google_definition_csv = '/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/definitions.xlsx'  
        function_list         = '/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/python_function_list.csv'  
        function_definition   = '/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/python_function_file_definition.csv"'  
        parameter_list        = '/Users/derekdewald/Documents/Python/Github_Repo/Streamlit/Data/python_function_parameters.csv'  
        daily_list            = '/Users/derekdewald/Documents/Python/Github_Repo/d_testing_folder/d_historical_test_results1.xlsx'
        
        data_dict['knowledge_base_df'] = pd.read_excel(knowledge_base_xlsx)
        data_dict['google_definition_df'] = pd.read_csv(google_definition_csv)
        data_dict['function_df'] = pd.read_csv(function_list)
        data_dict['function_def_df'] = pd.read_csv(function_definition)
        
        data_dict['parameter_df'] = pd.read_csv(parameter_list) 
        data_dict['daily_list_df'] = pd.read_excel(daily_list)
        
        print("Local Files Utilized for Knowledge, Consolidated, Process")

    except:
        knowledge_base_xlsx = "https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/knowledge_base.xlsx"
        google_definition_csv = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv'        
        function_list = "https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/python_function_list.csv"
        function_definition = "https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/python_function_file_definition.csv"
        parameter_list = "https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/python_function_parameters.csv"
        daily_list = "https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/d_testing_folder/d_historical_test_results1.xlsx"

        data_dict['knowledge_base_df'] = pd.read_excel(knowledge_base_xlsx)
        data_dict['google_definition_df'] = pd.read_csv(google_definition_csv)
        data_dict['function_df'] = pd.read_csv(function_list)
        data_dict['function_def_df'] = pd.read_csv(function_definition)
        data_dict['technical_notes_df'] = pd.read_csv(technical_notes)
        data_dict['parameter_df'] = pd.read_csv(parameter_list) 
        data_dict['daily_list_df'] = pd.read_excel(daily_list)

    
    # Normalize: keep your existing behavior (everything to string)
    for dict_key in data_dict.keys():
        for column in data_dict[dict_key].columns:
            data_dict[dict_key][column] = data_dict[dict_key][column].fillna("").astype(str)

    return data_dict    # Normalize: keep your existing behavior (everything to string)

data_dict = load_data()

# -----------------------
# Navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    [ "Home Page", 'Definitions','Processes, Taxonomy and Topics',"Technical Notes",'Functions',"Daily List"]#,'Summarization',"Knowledge Base",'ML Models']
     #"Frequency Summarization",,'Process Checklist',"Function List", "Function Parameters",  'Folder Table of Content', ]
)

if page == "Home Page":
    st.title("Derek's Data Science Knowledge Dasboard")
    st.markdown("""
    <ul>
        <li>The Bedrock of the dashboad is a series of google sheets, .py files maintained on my desktop (and saved in GIT) which represent the approach the processes I follow for work, development, and archival knowledge. The critical pieces are: 
            <ul>
                <li>Definitions</li>
                <li>Notes</li>
                <li>Knowledge Base</li>
                <li>Technical Notes</li>
                <li>Processes</li>
                <li>Processes Checklist</li>
                <li>Functions</li>
            </ul>
        </li>
        <li>Another Main Item</li>
    </ul>
    """, unsafe_allow_html=True)

# # -----------------------------------
# Definitions
# -----------------------------------

elif page == "Definitions":
    st.title("Definitions")

    df_base = data_dict["google_definition_df"].copy()

    # Only convert actual NaN/None to ""
    df_base = df_base.fillna("")

    required = ["Process", "Categorization", "Word", "Definition"]
    missing = [c for c in required if c not in df_base.columns]
    if missing:
        st.error(f"google_definition_df is missing required columns: {missing}")
        st.stop()

    # ----------------------------
    # 1) Slicers
    # ----------------------------
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        opts1 = ["(All)"] + sorted([x for x in df_base["Process"].astype(str).unique() if str(x).strip()])
        sel1 = st.selectbox("Process", opts1, index=0)

    df1 = df_base if sel1 == "(All)" else df_base[df_base["Process"].astype(str) == str(sel1)]

    with c2:
        opts2 = ["(All)"] + sorted([x for x in df1["Categorization"].astype(str).unique() if str(x).strip()])
        sel2 = st.selectbox("Categorization", opts2, index=0)

    df2 = df1 if sel2 == "(All)" else df1[df1["Categorization"].astype(str) == str(sel2)]

    with c3:
        opts3 = ["(All)"] + sorted([x for x in df2["Word"].astype(str).unique() if str(x).strip()])
        sel3 = st.selectbox("Word", opts3, index=0)

    df_view_full = df2 if sel3 == "(All)" else df2[df2["Word"].astype(str) == str(sel3)]
    st.caption(f"Rows: {len(df_view_full)}")

    # ----------------------------
    # 2) Grid (4 visible cols) + hidden _row_id
    # ----------------------------
    df_view_full = df_view_full.copy().reset_index(drop=False).rename(columns={"index": "_row_id"})


# Build Visual
    visible_cols = ["Process", "Categorization", "Word", "Definition"]
    grid_df = df_view_full[["_row_id"] + visible_cols].copy()

    gb = GridOptionsBuilder.from_dataframe(grid_df)

    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=True,
        autoHeight=True
    )

    gb.configure_selection("single", use_checkbox=False)
    gb.configure_column("_row_id", hide=True)

    gb.configure_column("Process", width=100, minWidth=80, maxWidth=120)
    gb.configure_column("Categorization", width=100, minWidth=80, maxWidth=120)
    gb.configure_column("Word", width=100, minWidth=80, maxWidth=120)

    gb.configure_column(
        "Definition",
        flex=1,
        minWidth=700,
        wrapText=True,
        autoHeight=True
    )


    gridOptions = gb.build()
    gridOptions["onGridReady"] = on_grid_ready
    gridOptions["onGridSizeChanged"] = on_grid_size_changed
    gridOptions["domLayout"] = "normal"

    grid_resp = AgGrid(
        grid_df,
        gridOptions=gridOptions,
        height=500,
        fit_columns_on_grid_load=False,
        reload_data=True,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        )



    selected_rows = grid_resp.get("selected_rows", [])
    if selected_rows is None:
        selected_rows = []
    elif isinstance(selected_rows, pd.DataFrame):
        selected_rows = selected_rows.to_dict("records")

    # ----------------------------
    # 3) Details (HTML-ish rendering)
    # ----------------------------
    st.subheader("Details")

    if len(selected_rows) == 0:
        st.info("Select a row above to view full details.")
    else:
        row_id = selected_rows[0].get("_row_id", None)

        if row_id is None:
            st.warning("Selection did not return _row_id (unexpected).")
        else:
            full_row = df_view_full[df_view_full["_row_id"] == row_id].head(1)

            if full_row.empty:
                st.warning("Could not locate the full record for the selected row.")
            else:
                rec = full_row.iloc[0].fillna("")

                # Show Image (if present)
                if "Image" in rec.index:
                    img_url = str(rec["Image"]).strip()
                    if img_url:
                        st.image(img_url, caption="Image", width=320)

                # Render each field/value like your reference function
                # (Exclude helper + Image since already shown)
                exclude_fields = {"_row_id", "Image"}

                for field, value in rec.items():
                    if field in exclude_fields:
                        continue

                    v = "" if value is None else str(value).strip()

                    # Always show field, even if blank
                    if field.lower() == "link":
                        if v:
                            st.markdown(f"**{field}:** [Open Link]({v})")
                        else:
                            st.markdown(f"**{field}:**")
                    elif field.lower() in {"markdown", "latex"}:
                        st.markdown(f"**{field}:**")
                        if v:
                            try:
                                st.latex(v)
                            except Exception:
                                st.write(v)
                        else:
                            st.write("")
                    else:
                        st.markdown(f"**{field}:**")
                        st.write(v)







# -----------------------------------
# Daily List
# -----------------------------------
elif page == 'Daily List':
    st.title("Daily Word, Definition, Quote List")

    df_base = data_dict['daily_list_df'].copy()

    for col in ['Date', 'Last Tested']:
        df_base[col] = pd.to_datetime(df_base[col],errors='coerce').dt.strftime('%Y-%m-%d')
  
    # Columns to display
    display_cols = [
        'Word',
        'Definition',
        'Score',
        'Date',
        'Last Tested',
        'Classification'
    ]

    # -------------------------
    # Classification Filter
    # -------------------------
    classification_sel = st.selectbox(
        "Filter",
        ["All", "Daily Requirements"],
        index=0
    )

    # -------------------------
    # Date Filter
    # -------------------------
    date_options = ["(All)"] + sorted(
        df_base['Date'].dropna().unique(),
        reverse=True
    )

    date_sel = st.selectbox(
        "Date",
        date_options,
        index=0
    )

    # -------------------------
    # Apply Filters
    # -------------------------
    df_view = df_base.copy()

    # Daily Requirements:
    # Exclude Concepts and Definitions
    if classification_sel == "Daily Requirements":
        df_view = df_view.loc[
            df_view['Classification'] != "Concepts and Definitions"
        ]

    # Apply date filter
    if date_sel != "(All)":
        df_view = df_view.loc[
            df_view['Date'] == date_sel
        ]

    # Keep only display columns
    df_view = df_view[display_cols].copy()

    # -------------------------
    # AG Grid
    # -------------------------
    gb = GridOptionsBuilder.from_dataframe(df_view)

    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=True,
        autoHeight=True
    )

    gb.configure_column('Word', width=120)
    gb.configure_column('Definition', flex=1, minWidth=400)
    gb.configure_column('Score', width=60)
    gb.configure_column('Date', width=70)
    gb.configure_column('Last Tested', width=70)
    gb.configure_column('Classification', width=100)

    gridOptions = gb.build()

    gridOptions["onGridReady"] = on_grid_ready
    gridOptions["onGridSizeChanged"] = on_grid_size_changed

    AgGrid(
        df_view,
        gridOptions=gridOptions,
        height=800,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        reload_data=True,
    )


# -----------------------------------
# Technical Notes
# -----------------------------------

elif page == 'Technical Notes':
    st.title("Technical Notes")
    df_base = data_dict['technical_notes_df'].copy()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    c1_word = 'Process'
    c2_word = 'Categorization'
    c3_word = 'Word'
    search_word = 'Definition'

    with c1:
        c1_options = ["(All)"] + sorted([x for x in df_base[c1_word].unique() if x.strip()])
        c1_sel = st.selectbox(c1_word, c1_options, index=0)

    df1 = df_base if c1_sel == "(All)" else df_base[df_base[c1_word] == c1_sel]

    with c2:
        c2_options = ["(All)"] + sorted([x for x in df1[c2_word].unique() if x.strip()])
        c2_sel = st.selectbox(c2_word, c2_options, index=0)

    df2 = df1 if c2_sel == "(All)" else df1[df1[c2_word] == c2_sel]

    with c3:
        c3_options = ["(All)"] + sorted([x for x in df2[c3_word].unique() if x.strip()])
        c3_sel = st.selectbox(c3_word, c3_options, index=0)

    df3 = df2 if c3_sel == "(All)" else df2[df2[c3_word] == c3_sel]

    with c4:
        definition_search = st.text_input("Definition search", value="", placeholder="Type to search Description...")

    df_view = df3
    if definition_search.strip():
        s = definition_search.strip().lower()
        df_view = df_view[df_view[search_word].str.lower().str.contains(s, na=False)]

    st.caption(f"Rows: {len(df_view)}")
    gb = GridOptionsBuilder.from_dataframe(df_view)

    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=True,
        autoHeight=True
    )

    gb.configure_column(c1_word, width=100, minWidth=80, maxWidth=120)
    gb.configure_column(c2_word, width=100, minWidth=80, maxWidth=120)
    gb.configure_column(c3_word, width=150, minWidth=120, maxWidth=170)

    gb.configure_column(
        search_word,
        flex=1,
        minWidth=700,
        wrapText=True,
        autoHeight=True
    )

    gridOptions = gb.build()

    gridOptions["onGridReady"] = on_grid_ready
    gridOptions["onGridSizeChanged"] = on_grid_size_changed

    AgGrid(
        df_view,
        gridOptions=gridOptions,
        height=800,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        reload_data=True,
    )


# -----------------------------------
# Functions
# -----------------------------------




elif page == 'Functions':
    st.title("Functions")
    df_base = data_dict['function_df'].copy()
    df_func_definition = data_dict['function_def_df'].copy()

  
    gb_def = GridOptionsBuilder.from_dataframe(df_func_definition)

    gb_def.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=True,
        autoHeight=True
    )

    gb_def.configure_column(
        "Function",
        flex=15,
        minWidth=120
    )

    gb_def.configure_column(
        "Definition",
        flex=85,
        minWidth=800
    )

    gridOptions_def = gb_def.build()

    AgGrid(
        df_func_definition,
        gridOptions=gridOptions_def,
        height=250,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        reload_data=True,
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    c1_word = 'Process'
    c2_word = 'Categorization'
    c3_word = 'Function'

    with c1:
        c1_options = ["(All)"] + sorted([x for x in df_base[c1_word].unique() if x.strip()])
        c1_sel = st.selectbox(c1_word, c1_options, index=0)

    df1 = df_base if c1_sel == "(All)" else df_base[df_base[c1_word] == c1_sel]

    with c2:
        c2_options = ["(All)"] + sorted([x for x in df1[c2_word].unique() if x.strip()])
        c2_sel = st.selectbox(c2_word, c2_options, index=0)

    df2 = df1 if c2_sel == "(All)" else df1[df1[c2_word] == c2_sel]

    with c3:
        c3_options = ["(All)"] + sorted([x for x in df2[c3_word].unique() if x.strip()])
        c3_sel = st.selectbox(c3_word, c3_options, index=0)

    df_view = df2 if c3_sel == "(All)" else df2[df2[c3_word] == c3_sel]
    df_view = df_view[['Folder','Function','Definition','Returns','Process','Categorization',]]

    


    st.caption(f"Rows: {len(df_view)}")
    gb = GridOptionsBuilder.from_dataframe(df_view)

    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=True,
        autoHeight=True
    )

    gb.configure_column("Folder", width=200, minWidth=180, maxWidth=200)
    gb.configure_column("Function", width=200, minWidth=180, maxWidth=200)
    gb.configure_column("Process", width=150, minWidth=140, maxWidth=150)
    gb.configure_column("Categorization", width=150, minWidth=140, maxWidth=150)
    gb.configure_column("Returns", width=150, minWidth=120, maxWidth=200)
    gb.configure_column("Definition", width=800, minWidth=700, maxWidth=750)

    gridOptions = gb.build()

    gridOptions["onGridReady"] = on_grid_ready
    gridOptions["onGridSizeChanged"] = on_grid_size_changed

    AgGrid(
        df_view,
        gridOptions=gridOptions,
        height=800,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        reload_data=True,
    )


# ---------------------------------------------------------
# Process, Taxonomy, and Topic
# ---------------------------------------------------------


elif page == "Processes, Taxonomy and Topics":
    st.title("Processes, Taxonomy and Topics")

    df_base = data_dict["knowledge_base_df"].copy()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    search_word = "Definition"

    # ---------------------------------------------------------
    # FILTER 1: Select Knowledge Structure
    # ---------------------------------------------------------
    with c1:
        structure_options = [
            "(Not Selected)",
            "Taxonomy",
            "Process",
            "Topic"
        ]

        structure_sel = st.selectbox(
            "Knowledge Structure",
            structure_options,
            index=0
        )

    # ---------------------------------------------------------
    # FILTER 2: Select Parent
    # ---------------------------------------------------------
    if structure_sel != "(Not Selected)":

        parent_df = df_base[
            df_base["Categorization"] == structure_sel
        ]

        parent_options = (
            parent_df["Word"]
            .dropna()
            .astype(str)
            .loc[lambda x: x.str.strip() != ""]
            .unique()
            .tolist()
        )

        parent_options = ["(All)"] + sorted(parent_options)

        with c2:
            parent_sel = st.selectbox(
                structure_sel,
                parent_options,
                index=0
            )

        # Filter using selected structure column
        if parent_sel == "(All)":
            df2 = df_base[
                df_base[structure_sel].notna()
                & (df_base[structure_sel].astype(str).str.strip() != "")
            ]
        else:
            df2 = df_base[
                df_base[structure_sel] == parent_sel
            ]

    else:
        parent_sel = "(All)"
        df2 = df_base

    # ---------------------------------------------------------
    # FILTER 3: Select Word
    # ---------------------------------------------------------
    with c3:

        word_options = (
            df2["Word"]
            .dropna()
            .astype(str)
            .loc[lambda x: x.str.strip() != ""]
            .unique()
            .tolist()
        )

        word_options = ["(All)"] + sorted(word_options)

        word_sel = st.selectbox(
            "Word",
            word_options,
            index=0
        )

    df3 = (
        df2
        if word_sel == "(All)"
        else df2[df2["Word"] == word_sel]
    )

    # ---------------------------------------------------------
    # FILTER 4: Definition Search
    # ---------------------------------------------------------
    with c4:
        definition_search = st.text_input(
            "Definition search",
            value="",
            placeholder="Type to search Definition..."
        )

    df_view = df3.copy()

    if definition_search.strip():
        s = definition_search.strip().lower()

        df_view = df_view[
            df_view[search_word]
            .astype(str)
            .str.lower()
            .str.contains(s, na=False)
        ]

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    st.caption(f"Rows: {len(df_view)}")

    gb = GridOptionsBuilder.from_dataframe(df_view)

    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=True,
        autoHeight=True
    )

    gb.configure_column(
        "Categorization",
        width=120,
        minWidth=100,
        maxWidth=150
    )

    gb.configure_column(
        "Word",
        width=180,
        minWidth=140,
        maxWidth=220
    )

    gb.configure_column(
        search_word,
        flex=1,
        minWidth=700,
        wrapText=True,
        autoHeight=True
    )

    gridOptions = gb.build()

    gridOptions["onGridReady"] = on_grid_ready
    gridOptions["onGridSizeChanged"] = on_grid_size_changed

    AgGrid(
        df_view,
        gridOptions=gridOptions,
        height=800,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        reload_data=True,
    )