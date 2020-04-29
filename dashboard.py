import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from PIL import Image

import time
# import plotly.express as px
# from skimage import io

# Loopoing over each image
df = pd.read_csv('simulation data/detection_results.csv')
images = []
# for i in range(1, max(df['nframe'])):
    # img = Image.open('simulation data/img{i}.jpg'.format(i))
    # images.append(img)
img = Image.open('simulation data/four_cones_raw.jpg')

fig = go.Figure(go.Image(z=img))
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
fig.update_layout(margin=dict(l=5, r=5, b=10,t=10))

color_mapping = {
                'blue':'#1F77B4',
                'orange':'#FF7F0E',
                'yellow':'#EECA3B'
}

colors = {
    'text': '#000000',
    'background': '#ffffff'
}


# fig1 = go.Figure(go.Image(z=img))
# fig1.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig1.update_layout(margin=dict(l=5, r=5, b=10,t=10))

# fig2 = go.Figure(go.Image(z=img))

# for i in range(len(df['u'])):
#     fig2.add_shape(
#         # unfilled Rectangle
#         type="rect",
#         x0=df['u'][i],
#         y0=df['v'][i],
#         x1=df['u'][i] + df['w'][i],
#         y1=df['v'][i] + df['h'][i],
#         line=dict(
#             color=color_mapping[df['type'][i]],
#         ),
#     )
#     fig2.add_annotation(
#         x=df['u'][i] + round(df['w'][i]/2),
#         y=df['v'][i],
#         text=f"ID:{i}",
#         bgcolor=colors['background'],
#         bordercolor=colors['text'],
#         opacity=0.8,
#         ax=1,
#         ay=-df['h'][i]/2,
#         align="center",
#         arrowsize=1,
#         arrowwidth=2,
#         arrowcolor=colors['text']    
#     )
    
# fig2.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig2.update_layout(margin=dict(l=5, r=5, b=10,t=10))

# fig3 = go.Figure(go.Image(z=img))

# for i in range(len(df['u'])):
    
#     fig3.add_annotation(
#         x=df['u'][i] + round(df['w'][i]/2),
#         y=df['v'][i],
#         text=f"X:{df['X'][i]}, Y:{df['Y'][i]}",
#         bgcolor='#ffffff',
#         bordercolor='#000000',
#         opacity=0.8,
#         ax=1,
#         ay=-df['h'][i]/2,
#         align="center",
#         arrowsize=1,
#         arrowwidth=2,
#         arrowcolor=colors['text'] 
            
#     )
    

# fig3.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig3.update_layout(margin=dict(l=5, r=5, b=10,t=10))

# figures = [fig1, fig2, fig3]





external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

                html.Div([

                    html.H1(
                        children='Perception',
                        style={
                            'textAlign': 'center',
                            'color': colors['text']
                        }
                    ),

                    html.Br(),

                    html.Div([
                            html.Div([
                                dcc.RadioItems(
                                    id='Left Image type',
                                    options=[
                                        {'label': 'Camera Image', 'value': 'camera'},
                                        {'label': 'Deapth Image', 'value': 'depth'},
                                        {'label': 'Lidar Image', 'value': 'Lidar'}],
                                    value='camera',
                                    # style={'width': '100%'},
                                    # style={"padding": "10px", "max-width": "800px", "margin": "auto"},
                                    labelStyle={'display': 'inline-block'}
                            )], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                            
                            dcc.Graph(
                                    id='Left perception_graph',
                                    figure=fig
                            ),

                            html.Div([html.H6('Image Display:')],style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                            
                            html.Div([
                                
                                dcc.Checklist(
                                            id='Left img_display',
                                            options=[
                                                {'label': 'Bounding Boxes   ', 'value': 'bb'},
                                                {'label': 'XYZ   ', 'value': 'XYZ'}],
                                            labelStyle={'display': 'inline-block'}
                                            # value='raw'
                                )],style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                            
                            # html.Hr()
                            
                    ],style={'width': '50%', 'float': 'left', 'display': 'inline-block',}),

                    html.Div([
                        
                            html.Div([
                                dcc.RadioItems(
                                    id='Right Image type',
                                    options=[
                                        {'label': 'Camera Image', 'value': 'camera'},
                                        {'label': 'Deapth Image', 'value': 'depth'},
                                        {'label': 'Lidar Image', 'value': 'Lidar'}],
                                    value='camera',
                                    # style={'width': '100%'},
                                    labelStyle={'display': 'inline-block'}
                            )], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                
                            dcc.Graph(
                                    id='Right perception_graph',
                                    figure=fig
                            ),
                            
                            html.Div([html.H6('Image Display:')],style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                            
                            html.Div([
                                
                                dcc.Checklist(
                                            id='Right img_display',
                                            options=[
                                                {'label': 'Bounding Boxes   ', 'value': 'bb'},
                                                {'label': 'XYZ   ', 'value': 'XYZ'}],
                                            labelStyle={'display': 'inline-block'}
                                            # value='raw'
                                )],style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                            
                            # html.Hr()
                    ],style={'width': '50%', 'float': 'right', 'display': 'inline-block',}),

                    html.Div([
                            
                            html.Div([
                                
                                html.Button('NEXT FRAME', id='button'),
                                
                                dcc.Input(
                                    id='frame input',
                                    placeholder='Enter frame number...',
                                    type='text',
                                    value=''
                                ),
                                
                                html.Button(id='submit button state', children='SUBMIT'),
                             ], style = {
                                        'width': '100%',
                                        'display': 'flex',
                                        'align-items': 'center',
                                        'justify-content': 'center',
                                        "padding": "20px", "max-width": "800px", "margin": "auto"
                                        }),
                            html.Div(
                                    id='Current frame state',
                                    children=html.H6('Current frame number is: 1'),
                                    style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),

                            html.Label('FPS'),
                            dcc.Slider(
                                min=0,
                                max=30,
                                step=None,
                                marks={
                                        0: '0',
                                        5: '0.5',
                                        10: '1',
                                        15: '2',
                                        20: '5',
                                        25: '15',
                                        30: '30'
                                },
                                value=0
                            ), 

                            html.Hr()

                    ])
                    # ,style={
                            # 'display': 'inline-block',
                            # 'width': '80%'})
                    
                ]),

                # html.Hr(),

                html.Div([
                    html.H1(
                        children='State Estimation',
                        style={
                            'textAlign': 'center',
                            'color': colors['text']
                        }
                    ),
                ])               
])


@app.callback(Output('Left perception_graph', 'figure'),
              [Input('Left img_display', 'value')])
def update_figure(selected_display):
   
    if selected_display == ['bb']:
        # Display image with bounding boxes
        fig = go.Figure(go.Image(z=img))

        for i in range(len(df['u'])):
            fig.add_shape(
                # unfilled Rectangle
                type="rect",
                x0=df['u'][i],
                y0=df['v'][i],
                x1=df['u'][i] + df['w'][i],
                y1=df['v'][i] + df['h'][i],
                line=dict(
                    color=color_mapping[df['type'][i]],
                ),
            )
            fig.add_annotation(
                x=df['u'][i] + round(df['w'][i]/2),
                y=df['v'][i],
                text=f"ID:{i}",
                bgcolor=colors['background'],
                bordercolor=colors['text'],
                opacity=0.8,
                ax=1,
                ay=-df['h'][i]/2,
                align="center",
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors['text']
            )

        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=5, r=5, b=10,t=10))

    elif selected_display == ['XYZ']:
        # Display image with 3D (x,y) cordinate
        fig = go.Figure(go.Image(z=img))

        for i in range(len(df['u'])):
           
            fig.add_annotation(
                x=df['u'][i] + round(df['w'][i]/2),
                y=df['v'][i],
                text=f"X:{df['X'][i]}, Y:{df['Y'][i]}",
                bgcolor='#ffffff',
                bordercolor='#000000',
                opacity=0.8,
                ax=1,
                ay=-df['h'][i]/2,
                align="center",
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors['text'] 
                    
            )
            
    
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=5, r=5, b=10,t=10))

    elif selected_display == ['bb', 'XYZ']:
        # Display image with bounding boxes and 3D (x,y) cordinate
        fig = go.Figure(go.Image(z=img))

        for i in range(len(df['u'])):
            fig.add_shape(
                # unfilled Rectangle
                type="rect",
                x0=df['u'][i],
                y0=df['v'][i],
                x1=df['u'][i] + df['w'][i],
                y1=df['v'][i] + df['h'][i],
                line=dict(
                    color=color_mapping[df['type'][i]],
                ),
            )
            fig.add_annotation(
                x=df['u'][i] + round(df['w'][i]/2),
                y=df['v'][i],
                text=f"ID:{i}, X:{df['X'][i]}, Y:{df['Y'][i]}",
                bgcolor='#ffffff',
                bordercolor=colors['text'],
                opacity=0.8,
                ax=1,
                ay=-df['h'][i]/2,
                align="center",
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors['text'] 
                    
            )
            
    
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=5, r=5, b=10,t=10))

    elif selected_display == ['XYZ', 'bb']:
        # Display image with bounding boxes and 3D (x,y) cordinate
        fig = go.Figure(go.Image(z=img))

        for i in range(len(df['u'])):
            fig.add_shape(
                # unfilled Rectangle
                type="rect",
                x0=df['u'][i],
                y0=df['v'][i],
                x1=df['u'][i] + df['w'][i],
                y1=df['v'][i] + df['h'][i],
                line=dict(
                    color=color_mapping[df['type'][i]],
                ),
            )
            fig.add_annotation(
                x=df['u'][i] + round(df['w'][i]/2),
                y=df['v'][i],
                text=f"ID:{i}, X:{df['X'][i]}, Y:{df['Y'][i]}",
                bgcolor='#ffffff',
                bordercolor=colors['text'],
                opacity=0.8,
                ax=1,
                ay=-df['h'][i]/2,
                align="center",
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors['text'] 
                    
            )
            
    
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=5, r=5, b=10, t=10))

    else:

        fig = go.Figure(go.Image(z=img))
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=5, r=5, b=10,t=10))

    return fig


@app.callback(Output('Right perception_graph', 'figure'),
              [Input('Right img_display', 'value')])
def update_figure(selected_display):
   
    if selected_display == ['bb']:
        # Display image with bounding boxes
        fig = go.Figure(go.Image(z=img))

        for i in range(len(df['u'])):
            fig.add_shape(
                # unfilled Rectangle
                type="rect",
                x0=df['u'][i],
                y0=df['v'][i],
                x1=df['u'][i] + df['w'][i],
                y1=df['v'][i] + df['h'][i],
                line=dict(
                    color=color_mapping[df['type'][i]],
                ),
            )
            fig.add_annotation(
                x=df['u'][i] + round(df['w'][i]/2),
                y=df['v'][i],
                text=f"ID:{i}",
                bgcolor=colors['background'],
                bordercolor=colors['text'],
                opacity=0.8,
                ax=1,
                ay=-df['h'][i]/2,
                align="center",
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors['text']
                
                    
            )
            
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=5, r=5, b=10,t=10))
   
    elif selected_display == ['XYZ']:
        # Display image with 3D (x,y) cordinate
        fig = go.Figure(go.Image(z=img))

        for i in range(len(df['u'])):
           
            fig.add_annotation(
                x=df['u'][i] + round(df['w'][i]/2),
                y=df['v'][i],
                text=f"X:{df['X'][i]}, Y:{df['Y'][i]}",
                bgcolor='#ffffff',
                bordercolor='#000000',
                opacity=0.8,
                ax=1,
                ay=-df['h'][i]/2,
                align="center",
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors['text']
                    
            )
            
    
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=5, r=5, b=10,t=10))

    elif selected_display == ['bb', 'XYZ']:
        # Display image with bounding boxes and 3D (x,y) cordinate
        fig = go.Figure(go.Image(z=img))

        for i in range(len(df['u'])):
            fig.add_shape(
                # unfilled Rectangle
                type="rect",
                x0=df['u'][i],
                y0=df['v'][i],
                x1=df['u'][i] + df['w'][i],
                y1=df['v'][i] + df['h'][i],
                line=dict(
                    color=color_mapping[df['type'][i]],
                ),
            )
            fig.add_annotation(
                x=df['u'][i] + round(df['w'][i]/2),
                y=df['v'][i],
                text=f"ID:{i}, X:{df['X'][i]}, Y:{df['Y'][i]}",
                bgcolor='#ffffff',
                bordercolor=colors['text'],
                opacity=0.8,
                ax=1,
                ay=-df['h'][i]/2,
                align="center",
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors['text']
                    
            )
            
    
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=5, r=5, b=10,t=10))

    elif selected_display == ['XYZ', 'bb']:
        # Display image with bounding boxes and 3D (x,y) cordinate
        fig = go.Figure(go.Image(z=img))

        for i in range(len(df['u'])):
            fig.add_shape(
                # unfilled Rectangle
                type="rect",
                x0=df['u'][i],
                y0=df['v'][i],
                x1=df['u'][i] + df['w'][i],
                y1=df['v'][i] + df['h'][i],
                line=dict(
                    color=color_mapping[df['type'][i]],
                ),
            )
            fig.add_annotation(
                x=df['u'][i] + round(df['w'][i]/2),
                y=df['v'][i],
                text=f"ID:{i}, X:{df['X'][i]}, Y:{df['Y'][i]}",
                bgcolor='#ffffff',
                bordercolor=colors['text'],
                opacity=0.8,
                ax=1,
                ay=-df['h'][i]/2,
                align="center",
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors['text']
                    
            )
            
    
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=5, r=5, b=10,t=10))

    else:

        fig = go.Figure(go.Image(z=img))
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=5, r=5, b=10,t=10))

    return fig


# # Perception Frames Controller - State input does not fire the callback
# app.callback(Output('output-state', 'children'),
#               [Input('submit-button-state', 'n_clicks')],
#               [State('input-1-state', 'value'))

# def update_output(n_clicks, input1, input2):
#     return u'''
#         The Button has been pressed {} times,
#         Input 1 is "{}",
#         and Input 2 is "{}"
#     '''.format(n_clicks, input1, input2)


if __name__ == '__main__':
    app.run_server(debug=True)
