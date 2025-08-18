# app.py
import io
import re
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Function Catalog Viewer", layout="wide")
st.title("📒 Function Catalog Viewer")

# ---- Fixed GitHub file (blob URL) ----
GITHUB_BLOB_URL = (
    "https://github.com/derek-dewald/Python_Tools/blob/main/"
    "d_py_functions/D_Python_Functions_17-Aug-25.xlsx"
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
    return resp.content  # bytes (pickle-safe)

@st.cache_data(show_spinner=False)
def read_first_sheet(xls_bytes: bytes) -> pd.DataFrame:
    with pd.ExcelFile(io.BytesIO(xls_bytes)) as xf:
        first = xf.sheet_names[0]
        df = xf.parse(sheet_name=first)
    # Parse date columns if present
    for c in ["Date Created", "Date Last Modified"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# ---- Load workbook into DataFrame
try:
    xls_bytes = fetch_workbook_bytes(RAW_URL)
    df = read_first_sheet(xls_bytes)
except Exception as e:
    st.error(f"Failed to load the workbook: {e}")
    st.stop()

# Preferred column order if present
preferred = [
    "File", "Function Name", "Description", "Parameters", "Returns", "Raises",
    "Examples", "Date Created", "Date Last Modified",
    "Return Type (Annotation)", "Start Line", "Code"
]
df = df[[c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]]

# Drop Code for a cleaner view (optional)
if "Code" in df.columns:
    df = df.drop(columns=["Code"])

st.divider()

# ---------- Filters (main page) ----------
st.subheader("Filters")

col1, col2, col3 = st.columns(3)

# Helper to build date options (YYYY-MM-DD strings)
def build_date_options(series: pd.Series) -> list[str]:
    if series.dtype.kind == "M":  # datetime-like
        vals = series.dropna().dt.date.astype(str).unique().tolist()
    else:
        # If they came in as strings, try normalizing
        s = pd.to_datetime(series, errors="coerce")
        vals = s.dropna().dt.date.astype(str).unique().tolist()
    vals = sorted(vals)
    return ["All", "Blank"] + vals

with col1:
    if "File" in df.columns:
        file_opts = ["All"] + sorted(df["File"].dropna().astype(str).unique().tolist())
        sel_file = st.selectbox("File", file_opts, index=0)
    else:
        sel_file = "All"

with col2:
    if "Date created" in df.columns:
        dc_opts = build_date_options(df["Date created"])
        sel_dc = st.selectbox("Date created", dc_opts, index=0)
    else:
        sel_dc = "All"

with col3:
    if "Date last modified" in df.columns:
        dlm_opts = build_date_options(df["Date last modified"])
        sel_dlm = st.selectbox("Date last modified", dlm_opts, index=0)
    else:
        sel_dlm = "All"


# Apply filters
view = df.copy()

if sel_file != "All" and "File" in view.columns:
    view = view[view["File"].astype(str) == sel_file]

def apply_date_filter(frame: pd.DataFrame, col: str, choice: str) -> pd.DataFrame:
    if col not in frame.columns or choice == "All":
        return frame
    if choice == "Blank":
        return frame[frame[col].isna() | (frame[col].astype(str).str.strip() == "")]
    # match YYYY-MM-DD by date
    if pd.api.types.is_datetime64_any_dtype(frame[col]):
        return frame[frame[col].dt.date.astype(str) == choice]
    # try to parse on the fly if not datetime
    col_parsed = pd.to_datetime(frame[col], errors="coerce")
    return frame[col_parsed.dt.date.astype(str) == choice]

if sel_dc != "All":
    view = apply_date_filter(view, "Date created", sel_dc)
if sel_dlm != "All":
    view = apply_date_filter(view, "Date last modified", sel_dlm)

# --- Text search filter (case-insensitive; matches any of these columns if present)
search_text = st.text_input("Search text", "")
if search_text:
    pattern = re.escape(search_text)
    searchable_cols = [c for c in [
        "Function Name", "Description", "Parameters", "Returns",
        "Raises", "Examples", "File"
    ] if c in view.columns]

    if searchable_cols:
        mask = pd.Series(False, index=view.index)
        for c in searchable_cols:
            mask |= view[c].astype(str).str.contains(pattern, case=False, na=False)
        view = view[mask]


st.divider()

# ---------- Wrapped table ----------
def wrap_df_for_display(df_in: pd.DataFrame) -> str:
    # Columns likely needing wrapping
    wrap_candidates = [c for c in ["Description", "Parameters", "Returns", "Raises", "Examples"] if c in df_in.columns]
    styles = []
    for col in wrap_candidates:
        # Header and cell wrapping
        styles.append(dict(selector=f"th.col_heading.level0#{col}", props=[("white-space", "normal")]))
        styles.append(dict(selector=f"td.col#{col}", props=[("white-space", "pre-wrap"), ("word-break", "break-word")]))
    # Wider fixed layout + slightly smaller font
    styles.append(dict(selector="table", props=[("table-layout", "fixed"), ("width", "100%")]))
    styles.append(dict(selector="tbody, thead", props=[("font-size", "0.9rem")]))

    styler = (
        df_in.style
        .hide(axis='index')
        .set_table_styles(styles, overwrite=False)
        .set_properties(**{"white-space": "pre-wrap"})  # fallback
    )
    return styler.to_html()

st.subheader("Results")
if view.empty:
    st.info("No rows match the selected filters.")
else:
    html_table = wrap_df_for_display(view)
    st.markdown(html_table, unsafe_allow_html=True)
