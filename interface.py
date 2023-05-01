import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objs as go
import pandas as pd
from colour import Color
from textwrap import dedent as d

from assets.compare_assets import get_filters_layout
from assets.home_assets import left_layout, right_layout
from dash.exceptions import PreventUpdate
from preprocessing import generate_graph
from faculty import filter_graph, Filter, compare_graphs

G = generate_graph()
pos = nx.spring_layout(G, k=0.5, iterations=20)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_csv('data/Faculty.csv')

def plot_graph(G, graph_name="Network"):
    traceRecode = []
    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                            hoverinfo="text", marker={'size': 30, 'color': 'LightSkyBlue'}, hoverlabel={"namelength" :-1})


    # adding nodes
    for node in G.nodes():
        x, y = pos[node]
        hovertext = "Name: " + str(G.nodes[node]['name']) + "<br>" + \
                    "Position: " + str(G.nodes[node]['position']) + "<br>" + \
                    "Gender: " + str(G.nodes[node]['gender']) + "<br>" + \
                    "Management: " + str(G.nodes[node]['mgmt']) + "<br>" + \
                    "Area: " + str(G.nodes[node]['area']) + "<br>" + \
                    "is_excellent: " + str(G.nodes[node]['is_excellent']) + "<br>" + \
                    "First Publication: " + str(G.nodes[node]['min_year'])
        text = str(G.nodes[node]['name'])
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([hovertext])
        node_trace['text'] += tuple([text])
    traceRecode.append(node_trace)

    if len(G.edges)>0:
        colors = list(Color('lightcoral').range_to(Color('darkred'), len(G.edges())))
        colors = ['rgb' + str(x.rgb) for x in colors]

        # adding edges
        index = 0
        for edge in G.edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = len(G[edge[0]][edge[1]]['papers']) / 8
            trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                               mode='lines',
                               line={'width': weight},
                               marker=dict(color=colors[index]),
                               line_shape='spline',
                               opacity=1)
            traceRecode.append(trace)
            index += 1

        # adding hovertext on center of edges
        middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
                                        marker={'size': 10, 'color': 'LightSkyBlue'}, opacity=0, hoverlabel={"namelength" :-1})
        for edge in G.edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            papers_text = ''
            papers_seen = set()
            for index, p in enumerate(G[edge[0]][edge[1]]['papers']):
                if str(p) not in papers_seen:
                    papers_text += str(index+1) + ". " + str(p) + "<br>"
                    papers_seen.add(str(p))
            hovertext = "From: " + str(G.nodes[edge[0]]['name']) + "<br>" + \
                        "To: " + str(G.nodes[edge[1]]['name']) + "<br>" +\
                        "Papers: " + "<br>" + papers_text

            middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
            middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
            middle_hover_trace['hovertext'] += tuple([hovertext])
        traceRecode.append(middle_hover_trace)

    figure = {
        "data": traceRecode,
        "layout": go.Layout(title= graph_name +  ' Visualization', showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 0, 'r': 0, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=800,
                            clickmode='event+select',
                            )}
    return figure

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'height': '300px',
        'overflowX': 'scroll',
        'overflowY': 'scroll',
        'padding-left': '10px'
    }
}

layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Div([html.H1("NTU SCSE DBLP Graph")],
                 className="row",
                 style={'textAlign': "center"}),

        html.Div(
            className="row",
            children=[
                left_layout,
                html.Div(
                    className="eight columns",
                    children=[dcc.Graph(id="my-graph",
                                        figure=plot_graph(G))], # set default account and year to the full range
                    style={'height': '800px'}
                ),
                right_layout
            ]
        ),],
    id='page-content'),
])

compare_layout = html.Div([
    html.Div(
        className="row",
        children=[
            get_filters_layout('g1'),
            html.Div(
                className="four columns",
                children=[dcc.Graph(id="graph1",
                                    figure=plot_graph(G, 'Graph 1'))], # set default account and year to the full range
                style={'height': '800px'}
            ),
            html.Div(
                className="four columns",
                children=[dcc.Graph(id="graph2",
                                    figure=plot_graph(G, 'Graph 2'))], # set default account and year to the full range
                style={'height': '800px'}
            ),
            get_filters_layout('g2'),
        ]
    ),
    html.Div(
        className="row",
        children=[dcc.Graph(id="graph3", figure=compare_graphs(G, G))], # set default account and year to the full range
        style={'height': '1000px'}
    ),
    dcc.Link('Go back to home', href='/'),
])

def get_app():
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.title = "NTU SCSE DBLP Network"
    app.layout = layout

    @app.callback(dash.dependencies.Output('my-graph', 'figure'),
                  [dash.dependencies.Input('start-year', 'value'), dash.dependencies.Input('end-year', 'value'),
                   dash.dependencies.Input('members', 'value'), dash.dependencies.Input('positions', 'value'),
                   dash.dependencies.Input('positions-1', 'value'), dash.dependencies.Input('genders', 'value'),
                   dash.dependencies.Input('mgmt', 'value'), dash.dependencies.Input('mgmt-1', 'value'),
                   dash.dependencies.Input('areas', 'value'), dash.dependencies.Input('excellent', 'value')])
    def update_output(start, end, members, positions, positions_1, genders, mgmt, mgmt_1, areas, excellence):
        start = 2000 if not start else start
        end = 2021 if not end else end
        period = [start, end]

        members = set(df.Faculty) if not members else members
        positions = set(df.Position) if not positions else {positions}
        positions_1 = set(df.Position) if not positions_1 else {positions_1}
        genders = set(df.Gender) if not genders else genders
        mgmt = set(df.Management) if not mgmt else {mgmt}
        mgmt_1 = set(df.Management) if not mgmt_1 else {mgmt_1}
        areas = set(df.Area) if not areas else areas
        excellence = {'True', 'False'} if not excellence else excellence

        filters = Filter(period=period, members=members, position_1=positions, position_2=positions_1, genders=genders,
                         mgmt_1=mgmt, mgmt_2=mgmt_1, areas=areas, is_excellent=excellence)
        G_ = filter_graph(G, filters)
        return plot_graph(G_)

    @app.callback(
        dash.dependencies.Output('hover-data', 'children'),
        [dash.dependencies.Input('my-graph', 'hoverData')])
    def display_hover_data(hoverData):
        if hoverData:
            return hoverData['points'][0]['hovertext'].replace('<br>', '\n')

    @app.callback(
        dash.dependencies.Output('click-data', 'children'),
        [dash.dependencies.Input('my-graph', 'clickData')])
    def display_click_data(clickData):
        if clickData:
            return clickData['points'][0]['hovertext'].replace('<br>', '\n')

    prev_pathname = ['/']
    @app.callback(dash.dependencies.Output('page-content', 'children'),
                  [dash.dependencies.Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/compare':
            prev_pathname[0] = '/compare'
            return compare_layout
        else:
            if prev_pathname[0] == '/compare':
                prev_pathname[0] = '/'
                return layout
            else:
                raise PreventUpdate

    @app.callback(dash.dependencies.Output('graph1', 'figure'),
                  [dash.dependencies.Input('start-year-g1', 'value'), dash.dependencies.Input('end-year-g1', 'value'),
                   dash.dependencies.Input('members-g1', 'value'), dash.dependencies.Input('positions-g1', 'value'),
                   dash.dependencies.Input('positions-1-g1', 'value'), dash.dependencies.Input('genders-g1', 'value'),
                   dash.dependencies.Input('mgmt-g1', 'value'), dash.dependencies.Input('mgmt-1-g1', 'value'),
                   dash.dependencies.Input('areas-g1', 'value'), dash.dependencies.Input('excellent-g1', 'value')])
    def get_graph_1(start, end, members, positions, positions_1, genders, mgmt, mgmt_1, areas, excellence):
        start = 2000 if not start else start
        end = 2021 if not end else end
        period = [start, end]

        members = set(df.Faculty) if not members else members
        positions = set(df.Position) if not positions else {positions}
        positions_1 = set(df.Position) if not positions_1 else {positions_1}
        genders = set(df.Gender) if not genders else genders
        mgmt = set(df.Management) if not mgmt else {mgmt}
        mgmt_1 = set(df.Management) if not mgmt_1 else {mgmt_1}
        areas = set(df.Area) if not areas else areas
        excellence = {'True', 'False'} if not excellence else excellence

        filters = Filter(period=period, members=members, position_1=positions, position_2=positions_1, genders=genders,
                         mgmt_1=mgmt, mgmt_2=mgmt_1, areas=areas, is_excellent=excellence)
        G_ = filter_graph(G, filters)
        return plot_graph(G_, 'Graph 1')

    @app.callback(dash.dependencies.Output('graph2', 'figure'),
                  [dash.dependencies.Input('start-year-g2', 'value'), dash.dependencies.Input('end-year-g2', 'value'),
                   dash.dependencies.Input('members-g2', 'value'), dash.dependencies.Input('positions-g2', 'value'),
                   dash.dependencies.Input('positions-1-g2', 'value'), dash.dependencies.Input('genders-g2', 'value'),
                   dash.dependencies.Input('mgmt-g2', 'value'), dash.dependencies.Input('mgmt-1-g2', 'value'),
                   dash.dependencies.Input('areas-g2', 'value'), dash.dependencies.Input('excellent-g2', 'value')])
    def get_graph_2(start, end, members, positions, positions_1, genders, mgmt, mgmt_1, areas, excellence):
        start = 2000 if not start else start
        end = 2021 if not end else end
        period = [start, end]

        members = set(df.Faculty) if not members else members
        positions = set(df.Position) if not positions else {positions}
        positions_1 = set(df.Position) if not positions_1 else {positions_1}
        genders = set(df.Gender) if not genders else genders
        mgmt = set(df.Management) if not mgmt else {mgmt}
        mgmt_1 = set(df.Management) if not mgmt_1 else {mgmt_1}
        areas = set(df.Area) if not areas else areas
        excellence = {'True', 'False'} if not excellence else excellence

        filters = Filter(period=period, members=members, position_1=positions, position_2=positions_1, genders=genders,
                         mgmt_1=mgmt, mgmt_2=mgmt_1, areas=areas, is_excellent=excellence)
        G_ = filter_graph(G, filters)
        return plot_graph(G_, 'Graph 2')

    @app.callback(dash.dependencies.Output('graph3', 'figure'),
                  [dash.dependencies.Input('start-year-g1', 'value'), dash.dependencies.Input('end-year-g1', 'value'),
                   dash.dependencies.Input('members-g1', 'value'), dash.dependencies.Input('positions-g1', 'value'), dash.dependencies.Input('positions-1-g1', 'value'),
                   dash.dependencies.Input('genders-g1', 'value'), dash.dependencies.Input('mgmt-g1', 'value'), dash.dependencies.Input('mgmt-1-g1', 'value'),
                   dash.dependencies.Input('areas-g1', 'value'), dash.dependencies.Input('excellent-g1', 'value'),
                   dash.dependencies.Input('start-year-g2', 'value'), dash.dependencies.Input('end-year-g2', 'value'),
                   dash.dependencies.Input('members-g2', 'value'), dash.dependencies.Input('positions-g2', 'value'), dash.dependencies.Input('positions-1-g2', 'value'),
                   dash.dependencies.Input('genders-g2', 'value'), dash.dependencies.Input('mgmt-g2', 'value'), dash.dependencies.Input('mgmt-1-g2', 'value'),
                   dash.dependencies.Input('areas-g2', 'value'), dash.dependencies.Input('excellent-g2', 'value')])
    def get_graph_3(start1, end1, members1, positions1, positions1_1, genders1, mgmt1, mgmt1_1, areas1, excellence1,
                    start2, end2, members2, positions2, positions2_1, genders2, mgmt2, mgmt2_1, areas2, excellence2,):
        start = 2000 if not start1 else start1
        end = 2021 if not end1 else end1
        period = [start, end]

        members = set(df.Faculty) if not members1 else members1
        positions = set(df.Position) if not positions1 else {positions1}
        positions_1 = set(df.Position) if not positions1_1 else {positions1_1}
        genders = set(df.Gender) if not genders1 else genders1
        mgmt = set(df.Management) if not mgmt1 else {mgmt1}
        mgmt_1 = set(df.Management) if not mgmt1_1 else {mgmt1_1}
        areas = set(df.Area) if not areas1 else areas1
        excellence = {'True', 'False'} if not excellence1 else excellence1

        filters = Filter(period=period, members=members, position_1=positions, position_2=positions_1, genders=genders,
                         mgmt_1=mgmt, mgmt_2=mgmt_1, areas=areas, is_excellent=excellence)
        G1 = filter_graph(G, filters)

        start = 2000 if not start2 else start2
        end = 2021 if not end2 else end2
        period = [start, end]

        members = set(df.Faculty) if not members2 else members2
        positions = set(df.Position) if not positions2 else {positions2}
        positions_1 = set(df.Position) if not positions2_1 else {positions2_1}
        genders = set(df.Gender) if not genders2 else genders2
        mgmt = set(df.Management) if not mgmt2 else {mgmt2}
        mgmt_1 = set(df.Management) if not mgmt2_1 else {mgmt2_1}
        areas = set(df.Area) if not areas2 else areas2
        excellence = {'True', 'False'} if not excellence2 else excellence2

        filters = Filter(period=period, members=members, position_1=positions, position_2=positions_1, genders=genders,
                         mgmt_1=mgmt, mgmt_2=mgmt_1, areas=areas, is_excellent=excellence)
        G2 = filter_graph(G, filters)

        return compare_graphs(G1, G2)

    return app

