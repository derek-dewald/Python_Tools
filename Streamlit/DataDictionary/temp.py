from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import streamlit as st
import random

st.set_page_config(
    page_title="Data Dictionary App",
    page_icon="📖",
    layout="wide",
)

# Load data from Google Sheets
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?gid=0&single=true&output=csv'
df = pd.read_csv(url)

# Text-based filter
st.subheader("Filter the Table")
search_query = st.text_input("Search:", placeholder="Type to search...")

# Columns to display in the AgGrid table
visible_columns = ['Word', 'Category', 'Sub Categorization']

# Apply filter
if search_query:
    filtered_df = df[
        df.apply(
            lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(),
            axis=1,
        )
    ][visible_columns]
else:
    filtered_df = df[visible_columns]

# Configure AgGrid options
builder = GridOptionsBuilder.from_dataframe(filtered_df)
builder.configure_default_column(
    wrapText=True,  # Enable text wrapping
    autoHeight=True,  # Adjust row height automatically
)
builder.configure_selection("single")  # Allow single row selection
grid_options = builder.build()

# Display the filtered table with AgGrid
st.subheader("Filtered Table with Row Selection")
response = AgGrid(
    filtered_df,
    gridOptions=grid_options,
    height=300,
    width="100%",
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,  # Required for advanced JavaScript features
)

selected_rows = response.get("selected_rows", [])
try:
    # Merge selected rows with the original DataFrame
    selected_data = pd.DataFrame(selected_rows)
    if not selected_data.empty:
        final_df = df.merge(selected_data[['Word']], on='Word', how='inner')


        # Display final_df with AgGrid
        st.subheader("Selected Row Details")
        detail_builder = GridOptionsBuilder.from_dataframe(final_df)
        detail_builder.configure_default_column(
            wrapText=True,  # Enable text wrapping
            autoHeight=True,  # Adjust row height automatically
        )
        detail_grid_options = detail_builder.build()

        AgGrid(
            final_df,
            gridOptions=detail_grid_options,
            height=300,
            width="100%",
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
        )
    else:
        st.write("No row selected.")
except Exception as e:
    st.write("An error occurred:", e)