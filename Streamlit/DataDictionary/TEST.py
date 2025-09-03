# --- Minimal Data Dictionary App ---

import streamlit as st
st.set_page_config(page_title="Data Dictionary", layout="wide")

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd
import numpy as np

# 🔗 Your Google Sheet (CSV export)
SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/"
    "pub?gid=0&single=true&output=csv"
)

# ---- Helpers ----
@st.cache_data(show_spinner=False)
def read_csv_clean(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # Normalize headers (strip + collapse spaces)
    df.columns = (
        pd.Index(df.columns)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add any missing columns as empty strings so the UI never crashes."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out

# ---- UI ----
with st.sidebar.expander("⚙️ Options", expanded=False):
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared – reload the page.")
    st.caption("Tip: if columns change in the Google Sheet, clear cache.")

st.title("🔍 Data Dictionary")

# Load and sanitize
df = read_csv_clean(SHEET_CSV_URL)

# Columns the app expects to show in the top grid
EXPECTED_LIST_COLS = ["Word", "Category", "Sub Categorization"]
df = ensure_cols(df, EXPECTED_LIST_COLS + ["Link", "Image", "Markdown"])

# Top search
st.subheader("Key Word / Phrase Search")
search_query = st.text_input("Search:", placeholder="Type to search...")

# Apply text search across all columns (case-insensitive)
if search_query:
    mask = df.apply(lambda r: r.astype(str).str.contains(search_query, case=False, na=False).any(), axis=1)
    list_df = df.loc[mask, EXPECTED_LIST_COLS].copy()
else:
    list_df = df[EXPECTED_LIST_COLS].copy()

# Reset index so we can hide it in AgGrid cleanly
list_df = list_df.reset_index(drop=False)

# Configure AgGrid
builder = GridOptionsBuilder.from_dataframe(list_df)
builder.configure_default_column(wrapText=True, autoHeight=True, cellStyle={'textAlign': 'center'})
builder.configure_selection("single", use_checkbox=False)
builder.configure_column("index", hide=True)
grid_options = builder.build()

st.subheader("📋 Key Terms and Classification")
grid_response = AgGrid(
    list_df,
    gridOptions=grid_options,
    height=320,
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
)

selected_rows = grid_response.get("selected_rows", [])
if not selected_rows:
    st.info("Select a row above to view details.")
else:
    # st_aggrid returns list[dict] for selected rows
    selected_df = pd.DataFrame(selected_rows)
    # Guard: make sure 'Word' exists (sheet/header changes won’t crash the app)
    if "Word" not in selected_df.columns:
        st.warning("Selection payload is missing the 'Word' column. Raw selection shown below:")
        st.dataframe(selected_df)
    else:
        # Join to get the full record for the selected word
        details = df.merge(selected_df[["Word"]], on="Word", how="inner")
        if details.empty:
            st.info("No matching details found for the selected word.")
        else:
            row0 = details.iloc[0].copy()
            transposed = row0.reset_index()
            transposed.columns = ["Field", "Value"]
            transposed["Value"] = transposed["Value"].fillna("")

            # Pretty-print details
            st.subheader("📑 Key Term Reference Details")
            for _, r in transposed.iterrows():
                field = str(r["Field"])
                value = r["Value"]

                if field == "Link":
                    if isinstance(value, str) and value.strip():
                        st.markdown(f"**{field}:** [Open Link]({value})")
                    else:
                        st.write(f"**{field}:**")
                elif field == "Image":
                    if isinstance(value, str) and value.strip():
                        st.image(value, caption="Image Reference", width=320)
                    else:
                        st.caption("No image available.")
                elif field == "Markdown":
                    # Treat as Markdown, not LaTeX (safer)
                    st.markdown(value or "")
                else:
                    st.write(f"**{field}:**\n{value}")

st.markdown("---")
st.markdown("🔗 [Open Raw Data in Google Sheets]"
            "(https://docs.google.com/spreadsheets/d/1tZ-_5Vv99_bm9CCEdDDN0KkmsFNcjWeKM86237yrCTQ/edit?gid=0#gid=0)")
