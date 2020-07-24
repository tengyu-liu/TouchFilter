import os
import sys
import datetime

import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output

import json

log_dir = os.path.join('logs/%s'%sys.argv[1])

app = dash.Dash(__name__)
app.layout = html.Div(
    html.Div([
        html.H4('Test'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )
    ])
)


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):
    print(n)
    fig = plotly.tools.json.load(open(os.path.join(log_dir, 'plotly.json')))
    print(fig)
    return fig


if __name__ == '__main__':

    app.run_server(debug=True)
    
