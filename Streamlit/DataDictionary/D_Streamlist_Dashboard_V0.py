import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
import datetime as dt
import textwrap
import html

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
    data_dict['d_learning_notes'] = data_dict['d_learning_notes'][['Category','Categorization','Word','Definition']]
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
    [ 'Words and Quotes','Daily Activities', "Function List", "Function Parameters", 'D Notes', 'D Definitions', 'Folder Table of Content', "D Notes Outline"]
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

# -----------------------------------
# Function Parameters
# -----------------------------------
elif page == "Function Parameters":
    st.title("Function Parameters")
    df_base = data_dict['parameter_list_df1'].copy()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    with c1:
        folder_opts = ["(All)"] + sorted([x for x in df_base["Folder"].unique() if x.strip()])
        sel_folder = st.selectbox("Folder", folder_opts, index=0)

    df1 = df_base if sel_folder == "(All)" else df_base[df_base["Folder"] == sel_folder]

    with c2:
        func_opts = ["(All)"] + sorted([x for x in df1["Function"].unique() if x.strip()])
        sel_func = st.selectbox("Function", func_opts, index=0)

    df2 = df1 if sel_func == "(All)" else df1[df1["Function"] == sel_func]

    with c3:
        param_opts = ["(All)"] + sorted([x for x in df2["Parameters"].unique() if x.strip()])
        sel_param = st.selectbox("Parameters", param_opts, index=0)

    df3 = df2 if sel_param == "(All)" else df2[df2["Parameters"] == sel_param]

    with c4:
        definition_search = st.text_input("Definition search", value="", placeholder="Type to search Definition...")

    df_view = df3
    if definition_search.strip():
        s = definition_search.strip().lower()
        df_view = df_view[df_view["Definition"].str.lower().str.contains(s, na=False)]

    st.caption(f"Rows: {len(df_view)}")

    gb = GridOptionsBuilder.from_dataframe(df_view)
    gb.configure_column("Folder", width=180)
    gb.configure_column("Function", width=220)
    gb.configure_column("Parameters", width=320)
    gb.configure_column("Definition", flex=1, wrapText=True, autoHeight=True)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)

    AgGrid(df_view, gridOptions=gb.build(), height=800, fit_columns_on_grid_load=True)

# -----------------------------------
# D Notes
# -----------------------------------
elif page == 'D Notes':
    st.title("D Notes")
    df_base = data_dict['google_notes_df'].copy()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    c1_word = 'Category'
    c2_word = 'Categorization'
    c3_word = 'Word'
    search_word = 'Description'

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
    gb.configure_column(c1_word, width=180)
    gb.configure_column(c2_word, width=220)
    gb.configure_column(c3_word, width=320)
    gb.configure_column(search_word, flex=1, wrapText=True, autoHeight=True)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)

    AgGrid(df_view, gridOptions=gb.build(), height=800, fit_columns_on_grid_load=True)

# -----------------------------------
# D Definitions
# -----------------------------------
elif page == "D Definitions":
    st.title("D Definitions")
    df_base = data_dict['google_definition_df'].copy()
    st.write(df_base)

# -----------------------------------
# Folder Table of Content
# -----------------------------------
elif page == "Folder Table of Content":
    st.title("Folder Table of Content")
    df_base = data_dict['folder_toc_df'].copy()
    if "Type" in df_base.columns:
        df_base.drop('Type', inplace=True, axis=1)
    st.write(df_base)

# -----------------------------------
# D Notes Outline
# -----------------------------------
elif page == "D Notes Outline":
    st.title("D Notes Outline")
    df_base = data_dict["d_learning_notes"].copy()

    c1_word = "Category"
    c2_word = "Categorization"
    c3_word = "Word"

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        opts1 = ["(All)"] + sorted([x for x in df_base[c1_word].unique() if x.strip()])
        sel1 = st.selectbox(c1_word, opts1, index=0)

    df1 = df_base if sel1 == "(All)" else df_base[df_base[c1_word] == sel1].drop(c1_word, axis=1)

    with c2:
        opts2 = ["(All)"] + sorted([x for x in df1[c2_word].unique() if x.strip()])
        sel2 = st.selectbox(c2_word, opts2, index=0)

    df2 = df1 if sel2 == "(All)" else df1[df1[c2_word] == sel2].drop(c2_word, axis=1)

    with c3:
        opts3 = ["(All)"] + sorted([x for x in df2[c3_word].unique() if x.strip()])
        sel3 = st.selectbox(c3_word, opts3, index=0)

    df_view = df2 if sel3 == "(All)" else df2[df2[c3_word] == sel3].drop(c3_word, axis=1)

    st.caption(f"Rows: {len(df_view)}")

    import streamlit.components.v1 as components
    html = notes_df_to_outline_html(df_view)
    components.html(html, height=800, scrolling=True)

# -----------------------------------
# Words and Quotes
# -----------------------------------
elif page == "Words and Quotes":
    st.title("Words and Quotes")
    df_base = data_dict["d_word_quote"].copy()
    df_base = df_base[(df_base['Text'].notnull()) & (df_base['Text'] != "")]

    # Sort by Date using the known format to avoid inference warnings
    if "Date" in df_base.columns:
        df_base["Date_sort"] = pd.to_datetime(df_base["Date"], format="%d-%b-%y", errors="coerce")
        df_base = df_base.sort_values("Date_sort", ascending=False).drop(columns=["Date_sort"])

    c1_word = "Date"
    c2_word = "Item"
    c3_word = "Source"
    c4_word = "Chapter"
    c5_word = "Verse(s)"
    search_word = "Text"

    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 2])

    with c1:
        opts1 = ["(All)"] + sorted([x for x in df_base[c1_word].unique() if str(x).strip()])
        sel1 = st.selectbox(c1_word, opts1, index=0)

    df1 = df_base if sel1 == "(All)" else df_base[df_base[c1_word] == sel1]

    with c2:
        opts2 = ["(All)"] + sorted([x for x in df1[c2_word].unique() if str(x).strip()])
        sel2 = st.selectbox(c2_word, opts2, index=0)

    df2 = df1 if sel2 == "(All)" else df1[df1[c2_word] == sel2]

    with c3:
        opts3 = ["(All)"] + sorted([x for x in df2[c3_word].unique() if str(x).strip()])
        sel3 = st.selectbox(c3_word, opts3, index=0)

    df3 = df2 if sel3 == "(All)" else df2[df2[c3_word] == sel3]

    with c4:
        opts4 = ["(All)"] + sorted([x for x in df3[c4_word].unique() if str(x).strip()])
        sel4 = st.selectbox(c4_word, opts4, index=0)

    df4 = df3 if sel4 == "(All)" else df3[df3[c4_word] == sel4]

    with c5:
        opts5 = ["(All)"] + sorted([x for x in df4[c5_word].unique() if str(x).strip()])
        sel5 = st.selectbox(c5_word, opts5, index=0)

    df5 = df4 if sel5 == "(All)" else df4[df4[c5_word] == sel5]

    with c6:
        text_search = st.text_input("Text search", value="", placeholder="Type to search Text...")

    df_view = df5
    if text_search.strip():
        s = text_search.strip().lower()
        df_view = df_view[df_view[search_word].str.lower().str.contains(s, na=False)]

    st.caption(f"Rows: {len(df_view)}")

    gb = GridOptionsBuilder.from_dataframe(df_view)
    gb.configure_column("Date", width=90)
    gb.configure_column("Item", width=90)
    gb.configure_column("Source", width=150)
    gb.configure_column("Chapter", width=90)
    gb.configure_column("Verse(s)", width=110)
    gb.configure_column("WP", width=110)
    gb.configure_column("Text", flex=1, wrapText=True, autoHeight=True)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)

    AgGrid(df_view, gridOptions=gb.build(), height=800, fit_columns_on_grid_load=True)

# -----------------------------------
# Daily Activities (UPDATED: correct window filtering via anchor date)
# -----------------------------------
elif page == "Daily Activities":
    st.title("Daily Activities")
    df = data_dict["daily_activities"].copy()

    # keep only rows with something in Bible (your prior guard)
    if "Bible" in df.columns:
        df = df[df['Bible'] != ""]

    if "Date" not in df.columns:
        st.error("daily_activities must include a 'Date' column.")
        st.stop()

    # Robust date parsing: try expected format first, then fallback
    dt1 = pd.to_datetime(df["Date"], format="%d-%b-%y", errors="coerce")
    dt2 = pd.to_datetime(df["Date"], errors="coerce")
    df["Date_dt"] = dt1.fillna(dt2)

    df = df.dropna(subset=["Date_dt"]).sort_values("Date_dt")
    if df.empty:
        st.warning("No valid rows after parsing Date. Check your 'Date' column format/content.")
        st.stop()

    act_cols = [c for c in df.columns if c not in ["Date", "Date_dt"]]

    for c in act_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # -------------------------
    # Controls
    # -------------------------
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        days_back = st.selectbox("Window", [7, 14, 30, 60, 90, 180, 365], index=2)

    with c2:
        view_mode = st.selectbox("Per-category chart type", ["Bars (0/1)", "Line (0/1)"], index=0)

    with c3:
        selected = st.multiselect("Categories", act_cols, default=act_cols)

    if not selected:
        st.warning("Select at least one category.")
        st.stop()

    # -------------------------
    # Anchor Date (NEW)
    # -------------------------
    max_dt = df["Date_dt"].max()
    max_date = max_dt.date() if pd.notna(max_dt) else dt.date.today()

    cA, cB = st.columns([1, 2])
    with cA:
        anchor_mode = st.selectbox("Anchor", ["Latest in data", "Today", "Pick a date"], index=0)

    if anchor_mode == "Latest in data":
        anchor_date = max_date
    elif anchor_mode == "Today":
        anchor_date = dt.date.today()
    else:
        with cB:
            anchor_date = st.date_input("Anchor date", value=max_date)

    anchor_ts = pd.Timestamp(anchor_date)
    min_dt = anchor_ts - pd.Timedelta(days=days_back - 1)

    dfw = df[(df["Date_dt"] >= min_dt) & (df["Date_dt"] <= anchor_ts)].copy()

    if dfw.empty:
        st.warning(
            f"No rows found between {min_dt.date()} and {anchor_ts.date()}. "
            f"Data range is {df['Date_dt'].min().date()} to {df['Date_dt'].max().date()}."
        )
        st.stop()

    st.caption(f"Showing: {min_dt.date()} → {anchor_ts.date()}  |  Rows: {len(dfw)}")

    # -------------------------
    # Daily totals
    # -------------------------
    dfw["Daily_Total"] = dfw[selected].sum(axis=1)
    dfw["Daily_Pct"] = (dfw["Daily_Total"] / len(selected)) * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest total", f"{int(dfw['Daily_Total'].iloc[-1])}/{len(selected)}")
    k2.metric(f"{days_back}-day avg total", f"{dfw['Daily_Total'].mean():.2f}/{len(selected)}")
    k3.metric("Latest %", f"{dfw['Daily_Pct'].iloc[-1]:.0f}%")
    k4.metric(f"{days_back}-day avg %", f"{dfw['Daily_Pct'].mean():.0f}%")

    fig_total = px.line(dfw, x="Date_dt", y="Daily_Total", markers=True, title="Daily Total Completed")
    st.plotly_chart(fig_total, use_container_width=True)

    dfw["Rolling_7D_Total"] = dfw["Daily_Total"].rolling(7, min_periods=1).sum()
    dfw["Rolling_30D_Total"] = dfw["Daily_Total"].rolling(30, min_periods=1).sum()

    fig_roll = px.line(
        dfw,
        x="Date_dt",
        y=["Rolling_7D_Total", "Rolling_30D_Total"],
        markers=False,
        title="Rolling Totals (7D / 30D)"
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    st.divider()

    # -------------------------
    # One graph per column
    # -------------------------
    st.subheader("Per-category charts")

    ncols = 2
    rows = [selected[i:i+ncols] for i in range(0, len(selected), ncols)]

    for row_cats in rows:
        cols = st.columns(ncols)
        for i, cat in enumerate(row_cats):
            with cols[i]:
                dcat = dfw[["Date_dt", cat]].copy()
                dcat["Rolling_7D_Rate"] = dcat[cat].rolling(7, min_periods=1).mean() * 100

                if view_mode.startswith("Bars"):
                    fig = px.bar(dcat, x="Date_dt", y=cat, title=f"{cat} (0/1)")
                else:
                    fig = px.line(dcat, x="Date_dt", y=cat, markers=True, title=f"{cat} (0/1)")

                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(
                    dcat, x="Date_dt", y="Rolling_7D_Rate",
                    markers=False, title=f"{cat} – 7D Completion Rate (%)"
                )
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # -------------------------
    # Weekly + Monthly summaries
    # -------------------------
    st.subheader("Weekly summary")

    dfw["Week_Start"] = dfw["Date_dt"].dt.to_period("W").dt.start_time

    weekly = (
        dfw.groupby("Week_Start")[selected]
        .sum()
        .reset_index()
    )
    weekly["Weekly_Total"] = weekly[selected].sum(axis=1)
    weekly["Weekly_Pct"] = (weekly["Weekly_Total"] / (len(selected) * 7)) * 100

    st.caption("Weekly_Pct assumes a 7-day week; partial weeks can look lower (expected).")
    fig_w = px.bar(weekly, x="Week_Start", y="Weekly_Total", title="Weekly Total Completed")
    st.plotly_chart(fig_w, use_container_width=True)
    st.dataframe(weekly, use_container_width=True)

    st.subheader("Monthly summary")

    dfw["Month_Start"] = dfw["Date_dt"].dt.to_period("M").dt.start_time

    monthly = (
        dfw.groupby("Month_Start")[selected]
        .sum()
        .reset_index()
    )

    days_in_month_present = dfw.groupby("Month_Start")["Date_dt"].nunique().reset_index(name="Days_Present")
    monthly = monthly.merge(days_in_month_present, on="Month_Start", how="left")

    monthly["Monthly_Total"] = monthly[selected].sum(axis=1)
    monthly["Monthly_Pct"] = (monthly["Monthly_Total"] / (len(selected) * monthly["Days_Present"])) * 100

    fig_m = px.bar(monthly, x="Month_Start", y="Monthly_Total", title="Monthly Total Completed")
    st.plotly_chart(fig_m, use_container_width=True)
    st.dataframe(monthly, use_container_width=True)
