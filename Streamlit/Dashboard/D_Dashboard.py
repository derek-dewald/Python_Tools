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
    google_note_csv = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSQF2lNc4WPeTRQ_VzWPkqSZp4RODFkbap8AqmolWp5bKoMaslP2oRVVG21x2POu_JcbF1tGRcBgodu/pub?output=csv'
    google_definition_csv = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv'
    knowledge_base_xlsx = "https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/knowledge_base.xlsx"
    technical_notes = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSnwd-zccEOQbpNWdItUG0qXND5rPVFbowZINjugi15TdWgqiy3A8eMRhbmSMBiRhHt1Qsry3E8tKY8/pub?output=csv'
    processes_xlsx = "https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/defined_processes.xlsx"
    consolidated_xlsx = "https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/consolidated_dataset.xlsx"
    function_list = "https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/python_function_list.csv"
    parameter_list = "https://raw.githubusercontent.com/derek-dewald/Python_Tools/main/Streamlit/Data/python_function_parameters.csv"

    data_dict = {}
    data_dict['google_notes_df'] = pd.read_csv(google_note_csv)
    data_dict['google_definition_df'] = pd.read_csv(google_definition_csv)
    data_dict['knowledge_base_df'] = pd.read_excel(knowledge_base_xlsx)
    data_dict['technical_notes_df'] = pd.read_csv(technical_notes)
    data_dict['processes_df'] = pd.read_excel(processes_xlsx)
    data_dict['consolidated_df'] = pd.read_excel(consolidated_xlsx)
    data_dict['function_df'] = pd.read_csv(function_list)
    data_dict['parameter_df'] = pd.read_csv(parameter_list)
    
    # Normalize: keep your existing behavior (everything to string)
    for dict_key in data_dict.keys():
        for column in data_dict[dict_key].columns:
            data_dict[dict_key][column] = data_dict[dict_key][column].fillna("").astype(str)

    return data_dict    # Normalize: keep your existing behavior (everything to string)
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
    [ "Home Page", 'Definitions','Notes',"Knowledge Base","Technical Notes",'Processes','Process and Categorization Utilization','Functions']
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
                <li>Processes and Categorization Utilization</li>
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
# Notes
# -----------------------------------
elif page == 'Notes':
    st.title("Notes")
    df_base = data_dict['google_notes_df'].copy()
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
# Knowledge Base
# -----------------------------------
elif page == 'Knowledge Base':
    st.title("Knowledge Base")
    df_base = data_dict['knowledge_base_df'].copy()
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
# Processes
# -----------------------------------

elif page == 'Processes':
    st.title("Processes")
    df_base = data_dict['processes_df'].copy()
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

    excel_bytes = df_to_excel_bytes(
        df_view,
        sheet_name="Processes",
        long_columns=["Definition"],
        default_max_width=30,
        long_max_width=80
    )

    st.download_button(
        label="Download filtered Processes as Excel",
        data=excel_bytes,
        file_name="filtered_processes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
# Process and Categorization Utilization
# -----------------------------------

elif page == 'Process and Categorization Utilization':
    st.title("Process and Categorization Utilization")
    df_base = data_dict['consolidated_df'].copy()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    c1_word = 'Process'
    c2_word = 'Categorization'
    c3_word = 'Location'

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
        default_index = (c3_options.index("Knowledge Base") if "Knowledge Base" in c3_options else 0)
        c3_sel = st.selectbox(c3_word,c3_options,index=default_index)

    df3 = df2 if c3_sel == "(All)" else df2[df2[c3_word] == c3_sel]

    df_view = df3[['Process','Categorization','Word','Definition']]
    gb_df = df3[['Process','Categorization','Location']].groupby(['Process','Categorization','Location']).size().reset_index().rename(columns={0:'Record Count'})

    st.caption(f"Rows: {len(gb_df)}")

    gb = GridOptionsBuilder.from_dataframe(gb_df)

    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=True,
        autoHeight=True
    )

    gb.configure_column(c1_word, width=180, minWidth=220, maxWidth=260)
    gb.configure_column(c2_word, width=140, minWidth=120, maxWidth=160)
    gb.configure_column('Location', width=140, minWidth=120, maxWidth=160)
    gb.configure_column('Record Count', width=100, minWidth=90, maxWidth=120)

    gridOptions = gb.build()

    gridOptions["onGridReady"] = on_grid_ready
    gridOptions["onGridSizeChanged"] = on_grid_size_changed

    table_col, blank_col = st.columns([5, 6])

    with table_col:
        AgGrid(
            gb_df,
            gridOptions=gridOptions,
            height= min(400, max(120, 45 + len(gb_df) * 35)),
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            reload_data=True,
        )

    gb1 = GridOptionsBuilder.from_dataframe(df_view)

    gb1.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        wrapText=True,
        autoHeight=True
    )

    gb1.configure_column(c1_word, width=100, minWidth=80, maxWidth=120)
    gb1.configure_column(c2_word, width=100, minWidth=80, maxWidth=120)
    gb1.configure_column('Word', width=150, minWidth=120, maxWidth=170)

    gb1.configure_column(
        "Definition",
        flex=1,
        minWidth=700,
        wrapText=True,
        autoHeight=True
    )

    gridOptions = gb1.build()

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
    df_view = df_view[['Folder','Function','Process','Categorization','Definition']]


    st.caption(f"Rows: {len(df_view)}")
    gb = GridOptionsBuilder.from_dataframe(df_view)

    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
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
