import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from PIL import Image


df = pd.read_csv('simulation data/detection_results.csv')
img = Image.open('simulation data/four_cones_raw.jpg')

display_options = [{'label': 'Raw image', 'value': 'raw'},
                   {'label': 'Bounding boxes', 'value': 'bb'},
                   {'label': 'Bounding boxes with types', 'value': 'bb+type'},
                   {'label': 'Bounding boxes with XYZ', 'value': 'bb+XYZ'}]
color_mapping = {'blue':'#1F77B4','orange':'#FF7F0E','yellow':'#EECA3B'}
app = dash.Dash()

app.layout = html.Div([
                html.Div([
                    dcc.Graph(id='perception_graph'),
                    dcc.Dropdown(id='img_display',options=display_options,value='raw')
                        ])
                ])

@app.callback(Output('perception_graph', 'figure'),
              [Input('img_display', 'value')])
def update_figure(selected_display):
    if selected_display == 'raw':
        # display only the image
        fig = go.Figure(go.Image(z=img))
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    elif selected_display == 'bb':
        # display image with bounding boxes
        fig = go.Figure(go.Image(z=img))
        for ind in range(len(df['u'])):
            fig.add_shape(
                # unfilled Rectangle
                type="rect",
                x0=df['u'][ind],
                y0=df['v'][ind],
                x1=df['u'][ind] + df['w'][ind],
                y1=df['v'][ind] + df['h'][ind],
                line=dict(
                    color="RoyalBlue",
                ),
            )
            fig.add_annotation(
                x=df['u'][ind] + round(df['w'][ind]/2),
                y=df['v'][ind],
                text=f"id: {ind}",
                bgcolor="RoyalBlue",
                opacity=0.6
            )


        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    elif selected_display == 'bb+type':
        #display image with bounding boxes and type
        fig = go.Figure(go.Image(z=img))
        for ind in range(len(df['u'])):
            fig.add_shape(
                # unfilled Rectangle
                type="rect",
                x0=df['u'][ind],
                y0=df['v'][ind],
                x1=df['u'][ind] + df['w'][ind],
                y1=df['v'][ind] + df['h'][ind],
                line=dict(
                    color=color_mapping[df['type'][ind]],
                ),

            )
            fig.add_annotation(
                x=df['u'][ind] + round(df['w'][ind]/2),
                y=df['v'][ind],
                text=f"id: {ind} type: {df['type'][ind]}",
                bgcolor=color_mapping[df['type'][ind]],
                opacity=0.6
            )
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    else:
        # display image with bounding boxes and XYZ
        # not yet implemented
        fig = go.Figure(go.Image(z=img))
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    return fig

if __name__ == '__main__':
    app.run_server()
