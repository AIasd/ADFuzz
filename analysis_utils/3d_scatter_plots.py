from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import os
import pandas
import numpy as np

# ----------------------- data parameters -----------------------
folder_path = 'no_simulation_dataset_script'
filename = 'grid.csv'

df = pandas.read_csv(os.path.join(folder_path, filename))

field_label_pairs = {
'x': 'ego_pos',
'y': 'ego_init_speed',
'z': 'other_pos',
'color': 'oob',
'size': 'other_init_speed',
'symbol': 'ped_delay'
}

# for each slide bar
max_step_size = 10

# ---------------------------------------------------------------

all_labels = list(field_label_pairs.values())
bar_values = []
for label in all_labels:
    values = df[label].to_numpy().astype('float')
    v_min = np.min(values)
    v_max = np.max(values)
    v_count = len(np.unique(values))
    v_count = np.min([max_step_size, v_count])
    bar_values.append((v_min, v_max, label, v_count))

sliders = []
for v_min, v_max, label, v_count in bar_values:
    step = (v_max-v_min) / v_count
    print(v_min, v_max, label, v_count, step)

    sliders.append(
    html.Div([
        html.P(label),
        dcc.RangeSlider(
            id='3d-scatter-plot-x-range-slider'+'-'+label,
            min=v_min, max=v_max, step=(v_max-v_min)/v_count,
            marks={int(v_min+i*step) if (v_min+i*step) % 1 == 0 else v_min+i*step: '{:.2f}'.format(v_min+i*step) for i in range(v_count+1)},
            value=[v_min, v_max]
    )], style={'width': '70%','padding-bottom':'1%', 'padding-left':'15%', 'padding-right':'15%'}))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H4('Data filtered by different features'),
    dcc.Graph(id="3d-scatter-plot-x-graph")]+sliders)
all_inputs = [Input('3d-scatter-plot-x-range-slider'+'-'+label, "value") for label in all_labels]

@app.callback(
    Output("3d-scatter-plot-x-graph", "figure"),
    all_inputs)
def update_bar_chart(*args):
    sliders_ranges = args

    mask = pandas.Series([True for _ in range(len(df.index))])
    for label, slider_range in zip(all_labels, sliders_ranges):
        v_min, v_max = slider_range
        mask = mask & (df[label] >= v_min) & (df[label] <= v_max)

    fig = px.scatter_3d(df[mask], size_max=18, opacity=0.7, **field_label_pairs)
    return fig

if __name__ == "__main__":
    app.run_server(port=9050, debug=True)
