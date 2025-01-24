from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Data Dictionary App",
    page_icon="📖",
    layout="wide",
)

# Load data from Google Sheets
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQq1-3cTas8DCWBa2NKYhVFXpl8kLaFDohg0zMfNTAU_Fiw6aIFLWfA5zRem4eSaGPa7UiQvkz05loW/pub?gid=0&single=true&output=csv'
df = pd.read_csv(url)

# Text-based filter
st.subheader("Key Word/ Phrase Search")
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
    cellStyle={'textAlign': 'center'}
)

builder.configure_selection("single")  # Allow single row selection
grid_options = builder.build()

# Display the filtered table with AgGrid
st.subheader("Key Terms and Classification")
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
        # Merge the selected row and mimic transposed view
        final_df = df.merge(selected_data[['Word']], on='Word', how='inner')
        transposed_df = pd.DataFrame({
            "Field": final_df.columns,
            "Value": final_df.iloc[0]
        }).fillna("")

        transposed_df["Value"] = transposed_df.apply(
            lambda row: f"[Open Link]({row['Value']})" if row["Field"] == "Link" and row["Value"] else row['Value'],  # Return blank if value is empty
            axis=1
            )
        
        transposed_df['Value'] = np.where(transposed_df['Value']=='[Open Link](nan)',"",transposed_df['Value'])
        

        st.subheader("Key Term Refernece Material")
        for _, row in transposed_df.iterrows():
            if row["Field"] == "Link":
                # Render the Link field as a markdown hyperlink
                st.markdown(f"**{row['Field']}:** {row['Value']}")
            else:
                st.write(f"**{row['Field']}:** {row['Value']}")
    else:
        st.write("No row selected.")
except Exception as e:
    st.write("An error occurred:", e)

st.markdown(
    """
    ### Relevant Links
    [Raw Data](https://docs.google.com/spreadsheets/d/1tZ-_5Vv99_bm9CCEdDDN0KkmsFNcjWeKM86237yrCTQ/edit?gid=0#gid=0)
    """
)