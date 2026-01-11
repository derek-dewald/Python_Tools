from __future__ import annotations

from st_aggrid import AgGrid, GridOptionsBuilder,GridUpdateMode
import streamlit.components.v1 as components
import streamlit as st

import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
import textwrap
import html


# To Download Project Template 
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
from typing import Iterable, Optional
from io import BytesIO


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



def notes_df_to_outline_html(
    df: pd.DataFrame,
    column_order=None):
    
    """

    Function to Take a Dataframe and convert it into A Structured Indented Point form Format. 
    Used for Clear Visualization of Notes.
    
    Parameters:
        df(df): Any DataFrame
        column_order(list): List of Columns to Include, in Order. If not defined, all will be included.
        print_(bool): Option as to whether you wish to directly Render a print out in the Python Session. Added because of Streamlit Error, need to suppress.

    Returns:
        str

    date_created:12-Dec-25
    date_last_modified: 18-Dec-25
    classification:TBD
    sub_classification:TBD
    usage:
        from connections import d_google_sheet_to_csv
        df = import_d_google_sheet('Notes')
        notes_df_to_outline_html(df)

    Update: 
        Added display parameter to support Streamlit Adoption.

    """
    if column_order is None:
        column_order = df.columns.tolist()

    missing = [c for c in column_order if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df1 = df[column_order].copy()

    def clean(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    last = [""] * len(column_order)

    html_ = """
    <style>
    .notes-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial; }
    .notes-item { line-height: 1.45; margin: 2px 0; }

    .notes-l0 { font-size: 18px; font-weight: 600; margin-left: 0px; }
    .notes-l1 { font-size: 16px; font-weight: 500; margin-left: 18px; }
    .notes-l2 { font-size: 14px; font-weight: 400; margin-left: 36px; }
    .notes-l3 { font-size: 13px; font-weight: 400; margin-left: 54px; opacity: 0.85; }
    .notes-l4 { font-size: 12px; font-weight: 400; margin-left: 72px; opacity: 0.8; }
    </style>

    <div class="notes-container">
    """

    for _, row in df1.iterrows():
        vals = [clean(row[c]) for c in column_order]
        if all(v == "" for v in vals):
            continue

        # Find first level where value changes
        change_level = None
        for i, v in enumerate(vals):
            if v and v != last[i]:
                change_level = i
                break

        # If nothing changes, show deepest non-blank value
        if change_level is None:
            for i in range(len(vals) - 1, -1, -1):
                if vals[i]:
                    change_level = i
                    break

        # Still nothing? (paranoia guard)
        if change_level is None:
            continue

        # Reset deeper levels when higher level changes (deeper only)
        for j in range(change_level + 1, len(last)):
            last[j] = ""

        # Render new values from change_level downward
        for i in range(change_level, len(vals)):
            v = vals[i]
            if not v:
                continue
            if v != last[i]:
                level = min(i, 4)  # cap style depth
                safe_v = html.escape(v) # Escape Function, not String.
                html_ += f'<div class="notes-item notes-l{level}">{safe_v}</div>\n'
                last[i] = v

    html_ += "</div>"
    html_ = textwrap.dedent(html_).lstrip()
    return html_

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
    function_list_url = (
        "https://raw.githubusercontent.com/"
        "derek-dewald/Python_Tools/main/Streamlit/DataDictionary/python_function_list.csv"
    )

    parameter_list_url = (
        "https://raw.githubusercontent.com/"
        "derek-dewald/Python_Tools/main/Streamlit/DataDictionary/python_function_parameters.csv"
    )

    folder_toc_url = (
        "https://raw.githubusercontent.com/"
        "derek-dewald/Python_Tools/main/Streamlit/DataDictionary/folder_listing.csv"
    )

    d_learning_notes_url = (
        "https://raw.githubusercontent.com/"
        "derek-dewald/Python_Tools/main/Streamlit/DataDictionary/d_learning_notes.csv"
    )

    google_note_csv = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSQF2lNc4WPeTRQ_VzWPkqSZp4RODFkbap8AqmolWp5bKoMaslP2oRVVG21x2POu_JcbF1tGRcBgodu/pub?output=csv'
    google_definition_csv = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv'
    google_word_quote = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTjXiFjpGgyqWDg9RImj1HR_BeriXs4c5-NSJVwQFn2eRKksitY46oJT0GvVX366LO-m1GM8znXDcBp/pub?gid=1117793378&single=true&output=csv'
    google_daily_activities = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTjXiFjpGgyqWDg9RImj1HR_BeriXs4c5-NSJVwQFn2eRKksitY46oJT0GvVX366LO-m1GM8znXDcBp/pub?gid=472900611&single=true&output=csv'

    data_dict = {}

    data_dict['google_notes_df'] = pd.read_csv(google_note_csv)
    data_dict['google_definition_df'] = pd.read_csv(google_definition_csv)
    data_dict['function_list_df'] = pd.read_csv(function_list_url)
    data_dict['parameter_list_df'] = pd.read_csv(parameter_list_url)
    data_dict['folder_toc_df'] = pd.read_csv(folder_toc_url)
    data_dict['d_learning_notes'] = pd.read_csv(d_learning_notes_url)
    data_dict['d_learning_notes'] = data_dict['d_learning_notes'][['Process','Categorization','Word','Definition']]
    data_dict['d_word_quote'] = pd.read_csv(google_word_quote)
    data_dict['daily_activities'] = pd.read_csv(google_daily_activities)

    # ✅ Folder first, then Function
    data_dict['function_list_df1']  = data_dict['function_list_df'][["Folder", "Function", "Purpose"]].copy()
    data_dict['parameter_list_df1'] = data_dict['parameter_list_df'][["Folder", "Function", "Parameters", "Definition"]].copy()

    # Normalize: keep your existing behavior (everything to string)
    for dict_key in data_dict.keys():
        for column in data_dict[dict_key].columns:
            data_dict[dict_key][column] = data_dict[dict_key][column].fillna("").astype(str)

    return data_dict

data_dict = load_data()

# -----------------------
# Navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    [ 'Words and Quotes','Daily Activities', "Function List", "Function Parameters", 'D Notes', 'D Definitions', 'Folder Table of Content', "D Notes Outline",'Project Template']
)

# -------------------------
# Function List
# -------------------------
if page == "Function List":
    st.title("Function List")
    df_base = data_dict['function_list_df1'].copy()
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        folder_opts = ["(All)"] + sorted([x for x in df_base["Folder"].unique() if x.strip()])
        sel_folder = st.selectbox("Folder", folder_opts, index=0)

    df1 = df_base if sel_folder == "(All)" else df_base[df_base["Folder"] == sel_folder]

    with c2:
        func_opts = ["(All)"] + sorted([x for x in df1["Function"].unique() if x.strip()])
        sel_func = st.selectbox("Function", func_opts, index=0)

    df2 = df1 if sel_func == "(All)" else df1[df1["Function"] == sel_func]

    with c3:
        purpose_search = st.text_input("Purpose search", value="", placeholder="Type to search Purpose...")

    df_view = df2
    if purpose_search.strip():
        s = purpose_search.strip().lower()
        df_view = df_view[df_view["Purpose"].str.lower().str.contains(s, na=False)]

    st.caption(f"Rows: {len(df_view)}")

    gb = GridOptionsBuilder.from_dataframe(df_view)
    gb.configure_column("Folder", width=100)
    gb.configure_column("Function", width=100)
    gb.configure_column("Purpose", flex=1, wrapText=True, autoHeight=True)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)

    AgGrid(df_view, gridOptions=gb.build(), height=800, fit_columns_on_grid_load=True)

