import dash_core_components as dcc
import dash_html_components as html
from textwrap import dedent as d
import pandas as pd

df = pd.read_csv('data/Faculty.csv')

def get_filters_layout(name):
    return html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            ** From: **
                            """)),
                            dcc.Dropdown(
                                id='start-year-' + name,
                                options=[
                                    {'label': i, 'value': i} for i in range(2000, 2022)
                                ],
                                value=None
                            ),
                            dcc.Markdown(d("""
                            ** To: **
                            """)),
                            dcc.Dropdown(
                                id='end-year-' + name,
                                options=[
                                    {'label': i, 'value': i} for i in range(2000, 2022)
                                ],
                                value=None
                            ),
                            html.Br(),
                        ],
                        style={'height': '160px'}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            ** Select members: **
                            """)),
                            dcc.Dropdown(
                                id="members-" + name,
                                options=[
                                    {'label': name, 'value': name} for name in df.Faculty
                                ],
                                value = [],
                                multi = True,
                            ),
                            html.Br(),
                            html.Div(id="output")
                        ],
                        style={'height': '100px'}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            ** Select positions: **
                            """)),
                            dcc.Dropdown(
                                id="positions-" + name,
                                options=[
                                    {'label': rank, 'value': rank} for rank in set(df.Position)
                                ],
                                value = [],
                                multi = False,
                            ),
                            html.Br(),
                            dcc.Dropdown(
                                id="positions-1-" + name,
                                options=[
                                    {'label': rank, 'value': rank} for rank in set(df.Position)
                                ],
                                value=[],
                                multi=False,
                            ),
                        ],
                        style={'height': '150px'}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            ** Select genders: **
                            """)),
                            dcc.Dropdown(
                                id="genders-" + name,
                                options=[
                                    {'label': gender, 'value': gender} for gender in set(df.Gender)
                                ],
                                value=[],
                                multi=True,
                            ),
                            html.Br(),
                        ],
                        style={'height': '100px'}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            ** Select management positions: **
                            """)),
                            dcc.Dropdown(
                                id="mgmt-" + name,
                                options=[
                                    {'label': str(mgmt), 'value': str(mgmt)} for mgmt in set(df.Management)
                                ],
                                value=[],
                                multi=False,
                            ),
                            html.Br(),
                            dcc.Dropdown(
                                id="mgmt-1-" + name,
                                options=[
                                    {'label': str(mgmt), 'value': str(mgmt)} for mgmt in set(df.Management)
                                ],
                                value=[],
                                multi=False,
                            ),
                        ],
                        style={'height': '150px'}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            ** Select research areas: **
                            """)),
                            dcc.Dropdown(
                                id="areas-" + name,
                                options=[
                                    {'label': area, 'value': area} for area in set(df.Area)
                                ],
                                value=[],
                                multi=True,
                            ),
                            html.Br(),
                        ],
                        style={'height': '100px'}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            ** Select excellence of members: **
                            """)),
                            dcc.Dropdown(
                                id="excellent-" + name,
                                options=[
                                    {'label': "Excellent", 'value': "True"},
                                    {'label': "Not Excellent", 'value': "False"},
                                ],
                                value=[],
                                multi=True,
                            ),
                            html.Br(),
                        ],
                        style={'height': '100px'}
                    ),
                ]
            )