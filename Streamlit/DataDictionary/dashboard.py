import streamlit as st
import plotly.express as px

# Streamlit app layout
st.title("Streamlit App")

# Example data
df = px.data.iris()

# Display Plotly chart
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
st.plotly_chart(fig)
