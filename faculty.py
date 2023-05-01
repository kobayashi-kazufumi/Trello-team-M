import networkx as nx
from preprocessing import generate_graph
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from copy import deepcopy
import pandas as pd

class Properties:
    def __init__(self, graph):
        self.graph = graph

    def info(self):
        # returns Number of Nodes, Edges and Average Degree
        info = nx.info(self.graph).split('\n')[2:]
        info = [i.split(' ')[-1] for i in info]
        return {'num_nodes': int(info[0]), 'num_edges': int(info[1]), 'average_degree': float(info[2]) if info[2]!='' else 0.0}

    def density(self):
        # returns the density of the Graph
        return nx.density(self.graph)

    def diameter(self):
        # returns the diameter of all connected components
        diameters = []
        connected_components = nx.connected_components(self.graph)
        for connected_component in connected_components:
            subgraph = self.graph.subgraph(connected_component)
            diameter = nx.diameter(subgraph)
            diameters.append(diameter)

        return diameters

    def average_path_length(self):
        # returns the average path length of all connected components
        average_path_lengths = []
        connected_components = nx.connected_components(self.graph)
        for connected_component in connected_components:
            subgraph = self.graph.subgraph(connected_component)
            average_path_length = nx.average_shortest_path_length(subgraph)
            average_path_lengths.append(average_path_length)

        return average_path_lengths

    def connected_components(self):
        # returns all the connected components 
        if nx.is_connected(self.graph):
            return self.graph
        connected_components = [(connected_component, nx.info(self.graph.subgraph(connected_component))) for connected_component in nx.connected_components(self.graph)]
        
        return connected_components

    def average_clustering_coefficient(self):
        # returns the average clustering coefficient
        return nx.average_clustering(self.graph)

    def degree_centrality(self):
        # returns the degree centrality
        return nx.degree_centrality(self.graph)

    def betweenness_centrality(self):
        # returns the betweeness centrality - https://en.wikipedia.org/wiki/Betweenness_centrality
        return nx.betweenness_centrality(self.graph, normalized=True, endpoints=True)

    def closeness_centrality(self):
        return nx.closeness_centrality(self.graph)

    def eigenvector_centrality(self):
        return nx.eigenvector_centrality(self.graph)

    def degree_distribution(self):
        return [self.graph.degree(n) for n in self.graph.nodes()]

    def shortest_paths_length(self):
        lengths = []
        n = len(self.graph.nodes())
        for i in range(n):
            for j in range(i+1, n):
                try:
                    lengths.append(nx.shortest_path_length(self.graph,source=i,target=j))
                except:
                    lengths.append(0) # no path found
        return lengths

    def connected_component_info(self):
        connected_components = nx.connected_components(self.graph)
        for connected_component in connected_components:
            subgraph = self.graph.subgraph(connected_component)
            print(nx.info(subgraph))
    
    def degree_distribution(self):
        return nx.degree_histogram(self.graph)

    def get_all_properties(self, verbose=0):
        # return a json to be displayed by front end
        properties_json = {}
        properties_json['info'] = self.info()
        properties_json['barplot_properties'] = {}
        properties_json['histogram_properties'] = {}
        properties_json['boxplot_properties'] = {}
        properties_json['scatterplot_properties'] = {}

        # barplots
        properties_json['barplot_properties']['num_nodes'] = properties_json['info']['num_nodes'] if self.graph.nodes() else 0
        properties_json['barplot_properties']['num_components'] = nx.number_connected_components(self.graph) if self.graph.nodes() else 0
        properties_json['barplot_properties']['num_edges'] = properties_json['info']['num_edges'] if self.graph.nodes() else 0
        properties_json['barplot_properties']['average_degree'] = properties_json['info']['average_degree'] if self.graph.nodes() else 0
        properties_json['barplot_properties']['density'] = self.density() if self.graph.nodes() else 0
        properties_json['barplot_properties']['average_clustering_coefficient'] = self.average_clustering_coefficient() if self.graph.nodes() else 0

        # histograms
        properties_json['histogram_properties']['diameter'] = self.diameter() if self.graph.nodes() else [0]
        properties_json['histogram_properties']['average_path_length'] = self.average_path_length() if self.graph.nodes() else [0]
        properties_json['histogram_properties']['shortest_paths_length'] = self.shortest_paths_length() if self.graph.nodes() else [0]

        # boxplots
        properties_json['boxplot_properties']['degree_centrality'] = self.degree_centrality() if self.graph.nodes() else [0]
        properties_json['boxplot_properties']['betweenness_centrality'] = self.betweenness_centrality() if self.graph.nodes() else [0]
        properties_json['boxplot_properties']['closeness_centrality'] = self.closeness_centrality() if self.graph.nodes() else [0]
        properties_json['boxplot_properties']['eigenvector_centrality'] = self.eigenvector_centrality() if self.graph.nodes() else [0]
        properties_json['connected_components'] = self.connected_components() if self.graph.nodes() else [0]

        # scatterplot
        properties_json['scatterplot_properties']['degree_distribution'] = self.degree_distribution() if self.graph.nodes() else [0]

        if verbose:
            print(properties_json['info'], end='\n\n')
            print("Density: ", properties_json['density'], end='\n\n')
            print("Diameter of connected components: ", properties_json['diameter'], end='\n\n')
            print("Average path length of connected components: ", properties_json['average_path_length'], end='\n\n')
            print("Average Clustering Coefficient: ", properties_json['average_clustering_coefficient'], end='\n\n')
            print("Degree Centrality: ", end='')
            print(properties_json['degree_centrality'], end='\n\n')
            print("Connected Components: ", end='')
            print(properties_json['connected_components'], end='\n\n')

        return properties_json

df = pd.read_csv('data/Faculty.csv')
class Filter:

    def __init__(self, members = set(df['Faculty']), position_1 = set(df['Position']), position_2 = set(df['Position']), genders = set(df['Gender']),
                       mgmt_1 = set(df['Management']), mgmt_2 = set(df['Management']), areas= set(df['Area']), is_excellent= {'True', 'False'},
                       period = (2000, 2021)):

        # filters for nodes
        self.members = members
        self.position = position_1.union(position_2)
        self.position_1 = position_1
        self.position_2 = position_2
        self.gender = genders
        self.mgmt = mgmt_1.union(mgmt_2)
        self.mgmt_1 = mgmt_1
        self.mgmt_2 = mgmt_2
        self.area = areas
        self.is_excellent = is_excellent

        # filter for edges

        # filter for both
        self.period = period


def filter_nodes(graph, filters):
    filtered_nodes = []
    for node in graph.nodes():
        if not(graph.nodes[node]['name'] in filters.members and graph.nodes[node]['position'] in filters.position and graph.nodes[node]['gender'] in filters.gender and graph.nodes[node]['mgmt'] in filters.mgmt \
                and graph.nodes[node]['area'] in filters.area and graph.nodes[node]['is_excellent'] in filters.is_excellent and filters.period[0]<=int(graph.nodes[node]['min_year'])<=filters.period[1]):
            filtered_nodes.append(node)
    return filtered_nodes

def filter_edges(graph, filters):
    filtered_edges= []
    for start,end in graph.edges():
        if not(any([filters.period[0]<=int(paper.year)<=filters.period[1] for paper in graph[start][end]['papers']])) or \
           not((graph.nodes[start]['position'] in filters.position_1 and graph.nodes[end]['position'] in filters.position_2) or
               (graph.nodes[end]['position'] in filters.position_1 and graph.nodes[start]['position'] in filters.position_2)) or \
           not((graph.nodes[start]['mgmt'] in filters.mgmt_1 and graph.nodes[end]['mgmt'] in filters.mgmt_2) or
                (graph.nodes[end]['mgmt'] in filters.mgmt_1 and graph.nodes[start]['mgmt'] in filters.mgmt_2)):
            filtered_edges.append((start, end))
    return filtered_edges

def filter_papers(graph, filters):
    for start,end in graph.edges():
        graph[start][end]['papers'] = [p for p in graph[start][end]['papers'] if filters.period[0]<=int(p.year)<=filters.period[1]]
    return graph

def filter_graph(graph, filters):
    filtered_nodes = filter_nodes(graph, filters)
    filtered_edges = filter_edges(graph, filters)
    graph_ = nx.restricted_view(graph, filtered_nodes, filtered_edges)
    return filter_papers(graph_, filters)

def compare_graphs(G1, G2, show=False):
    # get properties
    P1, P2 = Properties(G1), Properties(G2)
    P1_all, P2_all = P1.get_all_properties(), P2.get_all_properties()

    # layout of dashboard
    cols = 6
    rows = 5
    column_widths = [1/cols]*cols
    fig = make_subplots(
        rows=rows, cols=cols,
        column_widths=column_widths,
        row_heights=[0.1] + [(1-0.1)/(rows-1)]*(rows-1),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}]] +
              [[{"type": "histogram", "colspan": len(column_widths) // 3}, None, {"type": "histogram", "colspan": len(column_widths) // 3}, None, {"type": "histogram", "colspan": len(column_widths) // 3}, None]] +
              [[{"type": "box", "colspan":len(column_widths)//2}, None, None, {"type": "box", "colspan":len(column_widths)//2}, None, None]]*(rows-2))

    for index, p in enumerate(P1_all['barplot_properties']):
        v1 = P1_all['barplot_properties'][p]
        v2 = P2_all['barplot_properties'][p]

        fig.add_trace(go.Bar(x=['Graph 1','Graph 2'], y=[v1,v2], showlegend=False, marker_color=['steelblue', 'firebrick']),
                      row=1, col=index+1)
        fig.update_yaxes(title_text=p, row=1, col=index+1)

    for index, p in enumerate(P1_all['histogram_properties']):
        v1 = P1_all['histogram_properties'][p]
        v2 = P2_all['histogram_properties'][p]

        # fig.add_trace(go.Histogram(x=[v1,v2], showlegend=False, marker_color=['steelblue', 'firebrick']),
        #               row=2, col=2*index+1)
        fig.add_trace(go.Histogram(x=v1, name='Graph 1',  marker_color='steelblue'), row=2, col=2*index+1)
        fig.add_trace(go.Histogram(x=v2, name='Graph 2',  marker_color='firebrick'), row=2, col=2*index+1)
        fig.update_layout(
            bargap=0.2,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1,  # gap between bars of the same location coordinates
        )
        fig.update_yaxes(title_text=p, row=2, col=2*index+1)

    for index, p in enumerate(P1_all['boxplot_properties']):
        if isinstance(P1_all['boxplot_properties'][p], list):
            v1 = P1_all['boxplot_properties'][p]
        elif isinstance(P1_all['boxplot_properties'][p], dict):
            v1 = list(P1_all['boxplot_properties'][p].values())

        if isinstance(P2_all['boxplot_properties'][p], list):
            v2 = P2_all['boxplot_properties'][p]
        elif isinstance(P2_all['boxplot_properties'][p], dict):
            v2 = list(P2_all['boxplot_properties'][p].values())

        fig.add_trace(go.Box(name='Graph 1', y=v1, boxpoints='all', jitter=0.3, pointpos=-1.8, showlegend=False, marker_color='steelblue'),
                      row=index//2+3, col=(index%2*3)+1)
        fig.add_trace(go.Box(name='Graph 2', y=v2, boxpoints='all', jitter=0.3, pointpos=-1.8, showlegend=False, marker_color='firebrick'),
                      row=index//2+3, col=(index%2*3)+1)
        fig.update_yaxes(title_text=p, row=index//2+3, col=(index%2*3)+1)
    
    for index, p in enumerate(P1_all['scatterplot_properties']):
        v1 = P1_all['scatterplot_properties'][p]
        v2 = P2_all['scatterplot_properties'][p]
        degrees = max(list(range(len(v1))), list(range(len(v2))), key=len)
        fig.add_trace(go.Scatter(name="graph1", x=degrees, y=v1, mode='markers', showlegend=False, marker_color='steelblue'), row=index//2+5, col=(index%2*3)+1)
        fig.add_trace(go.Scatter(name="graph2", x=degrees, y=v2, mode='markers', showlegend=False, marker_color='firebrick'), row=index//2+5, col=(index%2*3)+4)
        fig.update_yaxes(title_text="degree distribution", row=index//2+5, col=(index%2*3)+1)
        fig.update_xaxes(title_text="degrees (Graph 1)", row=index//2+5, col=(index%2*3)+1)
        fig.update_yaxes(title_text="degree distribution", row=index//2+5, col=(index%2*3)+4)
        fig.update_xaxes(title_text="degrees (Graph 2)", row=index//2+5, col=(index%2*3)+4)

        # log axes
        fig.update_xaxes(type="log", row=index//2+5, col=(index%2*3)+1)
        fig.update_yaxes(type="log", row=index//2+5, col=(index%2*3)+1)
        fig.update_xaxes(type="log", row=index//2+5, col=(index%2*3)+4)
        fig.update_yaxes(type="log", row=index//2+5, col=(index%2*3)+4)
    fig.update_layout(
        title={
            'text': "Comparison between Graphs",
            'y': 0.97,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        autosize=True,
        # width=1900,
        height=2000,
    )

    if show:
        fig.show()

    return fig

if __name__ == "__main__":
    # retrieve the graph
    G = generate_graph()

    # generate properties of graph
    properties = Properties(G)
    properties.get_all_properties(verbose=0)

    # filter the graph
    filters = Filter(genders={'F'})
    G_ = filter_graph(G, filters)

    # compare 2 graphs
    compare_graphs(G, G_, show=True)
