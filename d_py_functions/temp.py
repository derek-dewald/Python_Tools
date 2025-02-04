import pandas as pd
import streamlit as st

# Example dataset
data = {
    "Category": ["Coconut", "Fruit", "Vegetable", "Vegetable", "Fruit"],
    "Item": ["Apple", "Banana", "Carrot", "Tomato", "Mango"],
    "Price": [1.2, 0.5, 0.8, 1.0, 1.5],
    "Description": [
        "A sweet red fruit",
        "A long yellow fruit",
        "A crunchy orange vegetable",
        "A juicy red vegetable",
        "A tropical sweet fruit",
    ],
}
df = pd.DataFrame(data)

# Sidebar filters
st.sidebar.header("Filter Options")

# Filter for "Category" column
category_filter = st.sidebar.multiselect(
    "Select Category:",
    options=df["Category"].unique(),
    default=df["Category"].unique(),  # Default: All categories selected
)

# Filter for "Item" column (searchable text input)
item_filter = st.sidebar.text_input("Search Item:")

# Filter for "Price" column (range slider)
price_filter = st.sidebar.slider(
    "Select Price Range:",
    min_value=float(df["Price"].min()),
    max_value=float(df["Price"].max()),
    value=(float(df["Price"].min()), float(df["Price"].max())),
)

# Apply filters dynamically
filtered_df = df[
    (df["Category"].isin(category_filter)) &
    (df["Item"].str.contains(item_filter, case=False, na=False)) &
    (df["Price"] >= price_filter[0]) &
    (df["Price"] <= price_filter[1])
]

# Display the filtered table
st.write("Filtered Data Table:")
st.dataframe(filtered_df, use_container_width=True)
