from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
import os
import pandas
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# ----------------------- fixed parameters (for now) -----------------------
# seed for deterministic result for SVC
random_seed = 0
# the list must be >= the number of distinct target values
color_map = ['blue', 'red', 'green', 'goldenrod', 'magenta']
symbol_map = SymbolValidator().values

use_subplots = True
# 2: 2d plot
vis_dim = 2
# ---------------------------------------------------------------

# ----------------------- data parameters -----------------------
folder_path = 'no_simulation_dataset_script'
filename = 'grid.csv'
df = pandas.read_csv(os.path.join(folder_path, filename))

# this line needs to be commented out to use the full dataset
df = df[:2000]

# if_draw_heatmap. It is supported only when vis_dim == 2.
if_draw_heatmap = True
# if show the scale bar on the right
showscale = True
# max number of steps for each slide bar
max_num_steps = 10

if use_subplots:
    # subplot_split_label is used to split subplots. It can be set to 'system version'.
    subplot_split_label = 'collision'
    # url_label is the url label and its address can be visited by clicking the corresponding point.
    # the url_label in field_label_pairs['customdata'] must be kept if it is used
    url_label = 'source'
    if vis_dim == 2:
        plot_f = go.Scattergl
        # marker_size and marker_symbol are optional 3rd and 4th dimensions
        # customdata is set to be a list of labels which are shown as mouse hover information shown up. 'data id' can be included as one element.

        field_label_pairs = {
        'x': 'ego_pos',
        'y': 'ego_init_speed',
        'marker_color': 'oob',
        # 'marker_size': 'other_pos',
        # 'marker_symbol': 'other_init_speed',
        'customdata': ['ego_pos', 'ego_init_speed', 'oob', 'other_init_speed', url_label]
        }
    # elif vis_dim == 3:
    #     plot_f = go.Scatter3d
    #     field_label_pairs = {
    #     'x': 'ego_pos',
    #     'y': 'ego_init_speed',
    #     'z': 'other_init_speed',
    #     'marker_color': 'oob',
    #     'text': 'other_pos'
    #     }
# else:
#     # size, symbol are all optional
#     # custom_data is usually set to be a list of labels which contain the information wanted to be shown but not used for plotting. The first label must be a number and it will be used to.
#     if vis_dim == 2:
#         field_label_pairs = {
#         'x': 'ego_pos',
#         'y': 'ego_init_speed',
#         'color': 'oob',
#         'size': 'other_pos',
#         'symbol': 'other_init_speed',
#         'custom_data': ['ped_delay', 'collision']
#         }
#         plot_f = px.scatter
#         additional_params = {}
#     elif vis_dim == 3:
#         field_label_pairs = {
#         'x': 'ego_pos',
#         'y': 'ego_init_speed',
#         'z': 'other_pos',
#         'color': 'oob',
#         'size': 'other_init_speed',
#         'symbol': 'ped_delay',
#         'custom_data': ['min_angle', 'collision']
#         }
#         plot_f = px.scatter_3d
#         additional_params = {}

# ---------------------------------------------------------------


all_labels = []
for k, v in field_label_pairs.items():
    if k not in ['text']:
        if k in ['custom_data', 'customdata']:
            for vi in v:
                if vi not in all_labels and vi not in [url_label]:
                    all_labels.append(vi)
        else:
            if v not in all_labels:
                all_labels.append(v)
bar_values = []
le_dict = {}
for label in all_labels:
    df_label_np = df[label].to_numpy()
    if df_label_np.dtype in ['int', 'float']:
        values = df_label_np.astype('float')
    else:
        le = LabelEncoder()
        le.fit(df_label_np)
        values = le.transform(df_label_np)
        le_dict[label] = le
    v_min = np.min(values)
    v_max = np.max(values)
    v_count = len(np.unique(values))
    v_count = np.min([max_num_steps, v_count])
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
    html.H1('Data filtered by different features', style={'textAlign': 'center'}),
    dcc.Graph(id="3d-scatter-plot-x-graph")]+sliders)
all_inputs = [Input('3d-scatter-plot-x-range-slider'+'-'+label, "value") for label in all_labels]


def get_bounds(df_x_label):
    x_min = np.min(df_x_label)
    x_max = np.max(df_x_label)
    x_span = x_max - x_min
    x_min -= x_span * 0.05
    x_max += x_span * 0.05
    return x_min, x_max

def draw_heatmap(df_mask_v, showscale, scaler, x_min, x_max, y_min, y_max):
    X = df_mask_v[[field_label_pairs['x'], field_label_pairs['y']]].to_numpy()
    y = df_mask_v[field_label_pairs['marker_color']].to_numpy()
    h = .02  # step size in the mesh

    if len(np.unique(y)) == 1:
        return False, None

    X = scaler.transform(X)
    clf = SVC(gamma=2, C=1, probability=True, random_state=random_seed)
    clf.fit(X, y)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_ = np.arange(y_min, y_max, h)

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,:1]
    Z = Z.reshape(xx.shape)

    return True, go.Heatmap(x=xx[0], y=y_, z=Z,
              colorscale='RdBu',
              showscale=showscale)
if url_label in field_label_pairs['customdata']:
    import webbrowser
    from dash.exceptions import PreventUpdate
    @app.callback(
        Output("3d-scatter-plot-x-graph", 'clickData'),
        [Input("3d-scatter-plot-x-graph", 'clickData')])
    def open_url(clickData):
        print('clickData', clickData)
        if clickData != None and 'customdata' in clickData['points'][0]:
            url_ind = field_label_pairs['customdata'].index(url_label)
            url = clickData['points'][0]['customdata'][url_ind]
            webbrowser.open_new_tab(url)
        else:
            raise PreventUpdate

@app.callback(
    Output("3d-scatter-plot-x-graph", "figure"),
    all_inputs)
def update_bar_chart(*args):
    sliders_ranges = args

    mask = pandas.Series([True for _ in range(len(df.index))])
    for label, slider_range in zip(all_labels, sliders_ranges):
        v_min, v_max = slider_range
        mask = mask & (df[label] >= v_min) & (df[label] <= v_max)
    df_mask = df[mask]

    if use_subplots:
        v_list = np.unique(df[subplot_split_label].to_numpy())
        num_subplots = len(v_list)
        subplot_titles = [subplot_split_label+'='+str(v) for v in v_list]
        col_num = int(np.ceil(np.sqrt(num_subplots)))
        row_num = int(np.ceil(num_subplots / col_num))
        fig = make_subplots(rows=row_num, cols=col_num, subplot_titles=subplot_titles)

        # Normalize x and y
        X_all = df[[field_label_pairs['x'], field_label_pairs['y']]].to_numpy()
        scaler = StandardScaler()
        scaler.fit(X_all)
        X_all_transformed = scaler.transform(X_all)

        # Get bounds
        x_min, x_max = get_bounds(X_all_transformed[:, 0])
        y_min, y_max = get_bounds(X_all_transformed[:, 1])


        for i, v in enumerate(v_list):
            row_i = i // col_num + 1
            col_i = i % col_num + 1
            df_mask_v = df_mask[df_mask[subplot_split_label]==v]
            print(subplot_split_label, '=', v, ', # data points =', len(df_mask_v))

            field_value_pairs = {}
            for field_name, label in field_label_pairs.items():
                if field_name not in ['x', 'y']:
                    value = df_mask_v[label].to_numpy()
                    if value.dtype not in ['int', 'float'] and field_name not in ['customdata', 'text']:
                        value = le_dict[label].transform(value)
                    if field_name == 'marker_color':
                        value = np.array([color_map[v] for v in value.astype('int')])
                    elif field_name == 'marker_symbol':
                        value = np.array([symbol_map[v] for v in value.astype('int')])
                    elif field_name == 'marker_size':
                        scaler_marker_size = MinMaxScaler()
                        scaler_marker_size.fit(np.expand_dims(df[label], 1))
                        value = np.squeeze(scaler_marker_size.transform(np.expand_dims(value, 1)))*20
                    field_value_pairs[field_name] = value
                    # click url test
                    # if field_name == 'customdata':
                    #     value[:, 0] = np.array(["https://www.google.com"]*value.shape[0])
                    #     field_value_pairs[field_name] = value


            X_transformed = scaler.transform(df_mask_v[[field_label_pairs['x'], field_label_pairs['y']]].to_numpy())
            field_value_pairs['x'] = X_transformed[:, 0]
            field_value_pairs['y'] = X_transformed[:, 1]

            if if_draw_heatmap:
                more_than_one_category, heatmap = draw_heatmap(df_mask_v, showscale, scaler, x_min, x_max, y_min, y_max)
                if more_than_one_category:
                    fig.append_trace(heatmap, row=row_i, col=col_i)

            hover_text_list = []
            for i, k in enumerate(field_label_pairs['customdata']):
                if k != url_label:
                    hover_text_list.append(k+': %{customdata['+str(i)+']}')
            hovertemplate = '<br>'.join(hover_text_list)

            fig.append_trace(go.Scatter(mode='markers', opacity=0.7, hovertemplate=hovertemplate, **field_value_pairs), row=row_i, col=col_i)
            fig.update_xaxes(title_text=field_label_pairs['x'], row=row_i, col=col_i)
            fig.update_yaxes(title_text=field_label_pairs['y'], row=row_i, col=col_i)

        # to keep the x and y ranges not changing while using slidebars.
        if vis_dim == 2:
            for i in range(1, num_subplots+1):
                fig.update_layout(**{'xaxis'+str(i)+'_range':[x_min, x_max], 'yaxis'+str(i)+'_range':[y_min, y_max]})
        fig.update_layout(showlegend=False)


    #     elif vis_dim == 3:
    #         for i in range(1, num_subplots+1):
    #             fig.update_layout(**{'xaxis'+str(i)+'_range':[x_min, x_max], 'yaxis'+str(i)+'_range':[y_min, y_max], 'zaxis'+str(i)+'_range':[z_min, z_max]})
    #     fig.update_layout(showlegend=False)
    #
    # else:
    #     fig = plot_f(df_mask, size_max=18, opacity=0.7, **additional_params, **field_label_pairs)
    #
    #     x_min, x_max = get_bounds(df[field_label_pairs['x']])
    #     y_min, y_max = get_bounds(df[field_label_pairs['y']])
    #
    #     if vis_dim == 2:
    #         fig.update_layout(xaxis_range=[x_min, x_max], yaxis_range=[y_min, y_max])
    #     elif vis_dim == 3:
    #         z_min, z_max = get_bounds(df[field_label_pairs['z']])
    #         fig.update_layout(xaxis_range=[x_min, x_max], yaxis_range=[y_min, y_max], zaxis_range=[z_min, z_max])
    #
    #     temp1 = fig.data[0].hovertemplate
    #     fig.update_traces(hovertemplate = temp1 + '<br>' + field_label_pairs['custom_data'][0] + "=%{customdata[0]}" + '<br>' + field_label_pairs['custom_data'][1] + "=%{customdata[1]}")

    return fig

if __name__ == "__main__":
    app.run_server(port=9050, debug=True)
