import pandas as pd
import networkx as nx
import requests
import xmltodict
import matplotlib.pyplot as plt
# import numpy as np
# from collections import OrderedDict

# for easier visualisation on front end
class Paper:

    def __init__(self, title, year, journal):
        self.title = title
        self.year = year
        self.journal = journal
        
    def __repr__(self):
        return self.title + " published in " + str(self.year) + " at " + self.journal

def get_link(link):
    response = requests.get(link)
    link = response.url.replace('.html', '.xml')
    return link

def get_pid(link):
    response = requests.get(link)
    data = xmltodict.parse(response.content)
    return data['dblpperson']['@pid']

def get_papers(link, df_data):
    response = requests.get(link)
    link = response.url.replace('.html', '.xml')

    response = requests.get(link)
    data = xmltodict.parse(response.content)

    if isinstance(data['dblpperson']['r'], list):
        for i in data['dblpperson']['r']:
            for k, v in i.items():
                if k == 'article':
                    other_pids = [elem['@pid'] for elem in v['author']] if isinstance(v['author'], list) else [v['author']['@pid']]
                    if 'ee' not in v:
                        df_data.append([data['dblpperson']['@pid'], other_pids, v['title'], v['year'], v['journal'], None, 'article'])
                    else:
                        df_data.append([data['dblpperson']['@pid'], other_pids, v['title'], v['year'], v['journal'], v['ee'], 'article'])
                elif k == 'inproceedings':
                    other_pids = [elem['@pid'] for elem in v['author']] if isinstance(v['author'], list) else [v['author']['@pid']]
                    if 'ee' not in v:
                        df_data.append([data['dblpperson']['@pid'], other_pids, v['title'], v['year'], v['booktitle'], None, 'inproceedings'])
                    else:
                        df_data.append([data['dblpperson']['@pid'], other_pids, v['title'], v['year'], v['booktitle'], v['ee'], 'inproceedings'])

    else:
        for k, v in data['dblpperson']['r'].items():
            if k == 'article':
                other_pids = [elem['@pid'] for elem in v['author']] if isinstance(v['author'], list) else [v['author']['@pid']]
                if 'ee' not in v:
                    df_data.append([data['dblpperson']['@pid'], other_pids, v['title'], v['year'], v['journal'], None, 'article'])
                else:
                    df_data.append([data['dblpperson']['@pid'], other_pids, v['title'], v['year'], v['journal'], v['ee'], 'article'])
            elif k == 'inproceedings':
                other_pids = [elem['@pid'] for elem in v['author']] if isinstance(v['author'], list) else [v['author']['@pid']]
                if 'ee' not in v:
                    df_data.append([data['dblpperson']['@pid'], other_pids, v['title'], v['year'], v['booktitle'], None, 'inproceedings'])
                else:
                    df_data.append([data['dblpperson']['@pid'], other_pids, v['title'], v['year'], v['booktitle'], v['ee'], 'inproceedings'])

def get_excellent(pid):
    tmp = papers[papers['pid'] == pid]
    return True in set(tmp['is_excellent']) if len(tmp)>0 else False

def get_min_year(pid):
    tmp = papers[papers['pid'] == pid]
    return min(tmp['year']) if len(tmp)>0 else 2000

def generate_graph():
    # loading required data
    global faculty
    global papers
    faculty = pd.read_csv('data/Faculty.csv')
    papers = pd.read_csv('data/Papers.csv')
    
    # get list of top venues in required format as seen when scrapping from HDLP
    top = pd.read_excel('data/Top.xlsx')
    top['HDLP_Venue'] = top['HDLP_Venue'].apply(eval)
    top = list(top.HDLP_Venue)
    top = set(sum(top, []))
    
    # check whether the paper is excellent
    is_excellent = [row['journal'] in top and row['type']=='inproceedings' for index, row in papers.iterrows()]
    papers['is_excellent'] = is_excellent

    # check whether paper in 2000 and after
    papers = papers[papers['year'] >= 2000]

    # check whether faculty member has produced an excellent paper and get the min date for visualisation across time
    faculty['is_excellent'] = faculty['pid'].apply(get_excellent)
    faculty['min_year'] = faculty['pid'].apply(get_min_year)

    # building the graph
    G = nx.Graph()
    nodes = {}
    
    # creating the nodes
    for index, row in faculty.iterrows():
        G.add_node(index, name=row['Faculty'], position=row['Position'], gender=row['Gender'], mgmt=row['Management'], area=row['Area'],
                   is_excellent=str(row['is_excellent']), min_year=row['min_year'], link=row['link'].replace('.xml', '.html'))
        nodes[row['pid']] = (index, G.nodes[index])
       
    # creating the edges
    for index, row in faculty.iterrows():
        tmp = papers[papers['pid'] == row['pid']]
        for i, p in tmp.iterrows():
            for pid in eval(p['other_pids']):
                if pid in nodes and pid != row['pid']:
                    n1, n2 = nodes[row['pid']][0], nodes[pid][0]
                    if G.has_edge(n1, n2):
                        G[n1][n2]['papers'].append(Paper(p['title'], p['year'], p['journal']))
                    else:
                        G.add_edge(n1, n2, papers=[Paper(p['title'], p['year'], p['journal'])])

    # visualising the graph
    nx.draw(G, with_labels=True, node_size=30, font_size=10, pos=nx.spring_layout(G, k=0.15, iterations=20))
    # plt.savefig('output/graph1.png', dpi=300, bbox_inches='tight')
    # plt.show()

    betCent = nx.betweenness_centrality(G, normalized=True, endpoints=True)
    node_color = [20000.0 * G.degree(v) for v in G]
    node_size =  [v * 10000 for v in betCent.values()]
    nx.draw(G, with_labels=True, node_size=node_size, node_color=node_color, font_size=10, pos=nx.spring_layout(G, k=0.15, iterations=20))
    # plt.savefig('output/graph2.png', dpi=300, bbox_inches='tight')

    return G

if __name__ == '__main__':
    generate_graph()
    