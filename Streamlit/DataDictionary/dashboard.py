from dash import Dash, dcc, html
import plotly.express as px

app = Dash(__name__)

# Example Data
df = px.data.iris()

# Create a Plotly figure
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

# Define app layout
app.layout = html.Div([
    html.H1("My Plotly Dashboard"),
    dcc.Graph(figure=fig)
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)