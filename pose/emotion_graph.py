import plotly.graph_objs as go
import pandas as pd
import plotly.express as px

def load_data(filepath):
    return pd.read_csv(filepath)

def filter_data(df, face_id):
    return df[df['Id'] == face_id]

def create_emotion_figure(face_df, emotions_to_plot):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, emotion in enumerate(emotions_to_plot):
        fig.add_trace(go.Scatter(
            x=face_df['Frame'],
            y=face_df[emotion],
            mode='lines',
            line=dict(width=2, color=colors[i % len(colors)]),
            name=emotion,
            hoverinfo='x+y+name',
            text=face_df['Frame'],
        ))

    enhance_figure_layout(fig)
    add_slider_to_figure(fig, face_df)
    return fig

def enhance_figure_layout(fig):
    fig.update_layout(
        title='Emotion Intensities Over Time for face_0',
        xaxis_title='Frame',
        yaxis_title='Intensity',
        hovermode='closest',
        template='plotly_dark',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10),
        )
    )

def add_slider_to_figure(fig, face_df):
    fig.update_layout(
        sliders=[{
            'currentvalue': {'prefix': 'Frame: ', 'visible': True, 'xanchor': 'center'},
            'steps': [{'method': 'animate', 'label': str(frame),
                       'args': [[{'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]]
                       } for frame in face_df['Frame']]
        }],
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]}
            ]
        }]
    )
