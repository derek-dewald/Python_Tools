from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import streamlit as st

# Load data from Google Sheets
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSnwd-zccEOQbpNWdItUG0qXND5rPVFbowZINjugi15TdWgqiy3A8eMRhbmSMBiRhHt1Qsry3E8tKY8/pub?output=csv'
df = pd.read_csv(url)

# Ensure Description column is clean and string-based
df['Description'] = df['Description'].fillna("").astype(str)

# Streamlit page configuration
st.set_page_config(
    page_title="Data Dictionary App",
    page_icon="📖",
    layout="wide",
)

st.title("DS Coding Dashboard")

# Create columns for filters on the main page
col1, col2, col3 = st.columns(3)

with col1:
    programs = ["All"] + sorted(df['Program'].unique().tolist())
    selected_program = st.selectbox("Select Program", programs)

with col2:
    classifications = ["All"] + sorted(df['Classification'].unique().tolist())
    selected_classification = st.selectbox("Select Classification", classifications)

with col3:
    descriptions = ["All"] + sorted(df['Description'].unique().tolist())
    selected_description = st.selectbox("Select Description", descriptions)

# Apply filters
filtered_df = df.copy()

if selected_program != "All":
    filtered_df = filtered_df[filtered_df['Program'] == selected_program]

if selected_classification != "All":
    filtered_df = filtered_df[filtered_df['Classification'] == selected_classification]

if selected_description != "All":
    filtered_df = filtered_df[filtered_df['Description'] == selected_description]

# Drop Program and Classification columns for display
display_df = filtered_df.drop(['Program', 'Classification'], axis=1)

# Configure AgGrid with centered text and enforced text wrapping
builder = GridOptionsBuilder.from_dataframe(display_df)

builder.configure_column('Command_Code', width=350, wrapText=True, suppressSizeToFit=True, cellStyle={'whiteSpace': 'normal', 'textAlign': 'center'})
builder.configure_column('Description', width=200, wrapText=True, suppressSizeToFit=True, cellStyle={'whiteSpace': 'normal', 'textAlign': 'center'})
builder.configure_column('Comments', width=800, wrapText=True, suppressSizeToFit=True, cellStyle={'whiteSpace': 'normal', 'textAlign': 'center'})


# Force text wrapping and center alignment
builder.configure_default_column(
    wrapText=True,
    autoHeight=True,
    resizable=True,  # Allow manual resizing
    cellStyle={      # Force wrapping via CSS
        'textAlign': 'center',
        'whiteSpace': 'normal',  # Critical for wrapping
        'wordBreak': 'break-word'  # Ensures long words wrap too
    }
)


# Build grid options
grid_options = builder.build()

# Display DataFrame using AgGrid
st.subheader("Filtered Results")
AgGrid(
    display_df,
    gridOptions=grid_options,
    fit_columns_on_grid_load=True,  # Prevent auto-fit to allow custom widths
    height=400,  # Fixed height ensures table renders correctly
    allow_unsafe_jscode=True
)

st.markdown(
    """
    ### Relevant Links
    [Raw Data](https://docs.google.com/spreadsheets/d/1FpYYq4LN6AZBaNRhnj1f76YNvnG-hTco40wJ1PUugto/edit?gid=0#gid=0)
    """
)