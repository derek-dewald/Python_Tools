


from st_aggrid import AgGrid, GridOptionsBuilder
import streamlit as st
import pandas as pd

st.set_page_config(page_title="AgGrid Row Selection Test")

st.title("AgGrid Selection Test")

data = pd.DataFrame({
    "Word": ["Apple", "Banana", "Cherry"],
    "Category": ["Fruit", "Fruit", "Fruit"]
}).reset_index(drop=True)

builder = GridOptionsBuilder.from_dataframe(data)
builder.configure_selection("single", use_checkbox=False)
builder.configure_grid_options(getRowNodeId="data.index")

grid_options = builder.build()

response = AgGrid(
    data,
    gridOptions=grid_options,
    allow_unsafe_jscode=True,
    update_mode='SELECTION_CHANGED',
    reload_data=True,
    height=300,
)

st.write("Selected row data:", response.get("selected_rows", []))