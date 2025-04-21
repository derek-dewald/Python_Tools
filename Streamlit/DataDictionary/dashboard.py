from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
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
st.subheader("Key Word/ Phrase Search - V2")
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

filtered_df = filtered_df.reset_index(drop=False)  # preserve index for row identification

# Build AgGrid options
builder = GridOptionsBuilder.from_dataframe(filtered_df)
builder.configure_default_column(
    wrapText=True,
    autoHeight=True,
    cellStyle={'textAlign': 'center'}
)
builder.configure_selection("single", use_checkbox=False)
builder.configure_column("index", hide=True)  # keep for selection reference
grid_options = builder.build()

# Render AgGrid
st.subheader("Key Terms and Classification")
response = AgGrid(
    filtered_df,
    gridOptions=grid_options,
    height=300,
    width="100%",
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    update_mode=GridUpdateMode.SELECTION_CHANGED  # critical fix
)

selected_rows = response.get("selected_rows", [])
st.write("✅ DEBUG Selected rows:", selected_rows)
st.write("🧾 First Word from DataFrame:", repr(df['Word'].iloc[0]))

try:
    # Merge selected row with full data
    selected_data = pd.DataFrame(selected_rows)
    if not selected_data.empty:
        final_df = df.merge(selected_data[['Word']], on='Word', how='inner')
        transposed_df = pd.DataFrame({
            "Field": final_df.columns,
            "Value": final_df.iloc[0]
        }).fillna("")

        transposed_df["Value"] = transposed_df.apply(
            lambda row: f"[Open Link]({row['Value']})" if row["Field"] == "Link" and row["Value"] else row['Value'],
            axis=1
        )
        transposed_df['Value'] = np.where(
            transposed_df['Value'] == '[Open Link](nan)', "", transposed_df['Value']
        )

        st.subheader("Key Term Reference Material")
        for _, row in transposed_df.iterrows():
            if row["Field"] == "Link":
                st.markdown(f"**{row['Field']}:** {row['Value']}")
            elif row["Field"] == "Image":
                if row["Value"] and not pd.isna(row["Value"]):
                    st.image(row["Value"], caption="Image Reference", width=300)
                else:
                    st.warning("No image available for this field.")
            elif row["Field"] == "Markdown":
                st.latex(row["Value"])
            else:
                st.write(f"**{row['Field']}:** \n{row['Value']}")
    else:
        st.write("No row selected.")
except Exception as e:
    st.error(f"An error occurred: {e}")

st.markdown(
    """
    ### Relevant Links
    [Raw Data](https://docs.google.com/spreadsheets/d/1tZ-_5Vv99_bm9CCEdDDN0KkmsFNcjWeKM86237yrCTQ/edit?gid=0#gid=0)
    """
)
