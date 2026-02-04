# streamlit_app.py
# Run: streamlit run streamlit_app.py
#
# What this does:
# - Loads your Google Sheet (published as CSV)
# - Lets you filter by Process and/or Categorization (single or multi-select)
# - Lets you choose which columns to GROUP BY (e.g., Process, Categorization, Learning Type, Model Type, etc.)
# - Shows a clean grouped count table (+ optional %)
# - (Optional) shows a bar chart of the grouped result

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?output=csv"

# Columns you said are categorical / semi-standard
DEFAULT_GROUP_COLS = ["Process", "Categorization", "Learning Type", "Algorithm Classification", "Model Type"]
DEFAULT_FILTER_COLS = ["Process", "Categorization"]

st.set_page_config(page_title="ML Definitions – Grouped Counts", layout="wide")
st.title("ML Definitions – Grouped Counts (Process / Categorization Washboard)")

@st.cache_data(show_spinner=False)
def load_df(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)

    # Basic cleanup: strip strings, normalize blanks -> NA
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = (
                df[c]
                .astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "N/A": pd.NA})
            )
    return df

df = load_df(CSV_URL)

with st.sidebar:
    st.header("Filters")

    # Confirm available columns
    cols = df.columns.tolist()
    proc_col = "Process" if "Process" in cols else None
    cat_col  = "Categorization" if "Categorization" in cols else None

    if proc_col:
        proc_vals = sorted([x for x in df[proc_col].dropna().unique()])
        sel_proc = st.multiselect("Process", proc_vals, default=[])
    else:
        sel_proc = []
        st.warning("Column 'Process' not found in the dataset.")

    if cat_col:
        cat_vals = sorted([x for x in df[cat_col].dropna().unique()])
        sel_cat = st.multiselect("Categorization", cat_vals, default=[])
    else:
        sel_cat = []
        st.warning("Column 'Categorization' not found in the dataset.")

    st.divider()
    st.header("Group By")

    # Let you pick exactly which dimensions to group by
    candidate_group_cols = [c for c in DEFAULT_GROUP_COLS if c in cols]
    # Also allow Word if you want it (often huge; optional)
    if "Word" in cols:
        candidate_group_cols = candidate_group_cols + ["Word"]

    group_cols = st.multiselect(
        "Choose columns to group by",
        options=candidate_group_cols,
        default=[c for c in ["Process", "Categorization"] if c in candidate_group_cols],
        help="Pick 1–3 columns for the clearest output.",
    )

    show_percent = st.checkbox("Show percent of filtered rows", value=True)
    include_na = st.checkbox("Include <NA> groups", value=False)

    st.divider()
    st.header("Chart (optional)")
    show_chart = st.checkbox("Show bar chart for grouped result", value=True)
    top_n = st.slider("Top N bars", 5, 50, 20, 5)

# Apply filters
df_f = df.copy()

if proc_col and sel_proc:
    df_f = df_f[df_f[proc_col].isin(sel_proc)]

if cat_col and sel_cat:
    df_f = df_f[df_f[cat_col].isin(sel_cat)]

st.caption(f"Rows (after filters): **{len(df_f):,}**  |  Total rows: **{len(df):,}**")

# Guardrails
if not group_cols:
    st.info("Pick at least one **Group By** column in the sidebar.")
    st.stop()

# Grouped counts
gb = df_f[group_cols].copy()

if not include_na:
    # Drop rows where ANY group-by column is NA
    gb = gb.dropna(subset=group_cols, how="any")
else:
    # Fill NA labels so they appear explicitly
    for c in group_cols:
        gb[c] = gb[c].fillna("<NA>")

if gb.empty:
    st.warning("No rows left after applying filters and NA-handling. Try including <NA> or loosening filters.")
    st.stop()

result = (
    gb
    .groupby(group_cols, dropna=False)
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
    .reset_index(drop=True)
)

if show_percent:
    denom = result["Count"].sum()
    result["Percent"] = (result["Count"] / denom * 100).round(2)

st.subheader("Grouped Counts")
st.dataframe(result, use_container_width=True)

# Optional bar chart (works best when grouping by 1–2 cols)
if show_chart:
    st.subheader("Bar Chart (Top Groups)")

    chart_df = result.head(top_n).copy()

    # Create a label for multi-column grouping
    if len(group_cols) == 1:
        chart_df["Label"] = chart_df[group_cols[0]].astype(str)
    else:
        chart_df["Label"] = chart_df[group_cols].astype(str).agg(" | ".join, axis=1)

    # Plot
    fig = plt.figure(figsize=(10, max(3, 0.35 * len(chart_df) + 1)))
    ax = fig.add_subplot(111)

    # Reverse for top-to-bottom readability
    chart_df = chart_df.iloc[::-1]
    ax.barh(chart_df["Label"], chart_df["Count"])
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.set_title(f"Top {len(chart_df)} grouped values")

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

with st.expander("Download grouped result"):
    st.download_button(
        "Download CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="grouped_counts.csv",
        mime="text/csv",
    )
