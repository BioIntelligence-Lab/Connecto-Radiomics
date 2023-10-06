'''
@author ichaudry
run the connectomics on the radiomics results
'''

#imports
import pandas as pd
import networkx as nx
from tqdm import tqdm
from pyvis import network as net
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import itertools
import copy
import sys
import traceback

from read_parameters import parse_params


#distance calculation functions
def eucl_distance(x, y):
    '''
    Calculates and returns the euclidian distance between x and y.

    Parameters:
    x,y : pandas series that each represent the radiomic profile of a single feature from the same patient.

    Returns:
    dist : the distance value 
    '''

    dist = math.dist(x, y)
    return dist

def cosine_dist(x, y):
    '''
    Calculates and returns the cosine distance between x and y.

    Parameters:
    x,y : pandas series that each represent the radiomic profile of a single feature from the same patient.

    Returns:
    dist : the distance value 
    '''

    dist = np.dot(x,y)/(norm(x)*norm(y))
    return dist

#function to process a single patient(i.e. claculating all edge weights and making network)
def process_image(img_id, distance_function=eucl_distance):
    ''' 
    Generates the connectomics graph for an image/scan.

    Parameters:
    img_id: str
        The ID for the target img_id. 
    
    distance_function:
        The distance function to use to make the graph

    Returns:
    network: networkx object
    '''
    edgelist = pd.DataFrame({
        'Source':[],
        'Target':[],
        'Weight':[]
    })


    for i, j in list(itertools.combinations(FEATURE_DICT.keys(), 2)):

        feature1_data = all_data[int(i)-1]
        feature2_data = all_data[int(j)-1]

        
        feature1_vector = feature1_data[feature1_data['img_id'] == img_id].values[0][1:]
        feature2_vector = feature2_data[feature1_data['img_id'] == img_id].values[0][1:]

        row = {
            'Source':[FEATURE_DICT[i]],
            'Target':[FEATURE_DICT[j]],
            'Weight':[distance_function(feature1_vector, feature2_vector)]
            }
        
        edgelist = pd.concat([edgelist, pd.DataFrame(row)])
    
    network = nx.from_pandas_edgelist(df=edgelist,
                                       source='Source',
                                       target='Target',
                                       edge_attr='Weight')
    
    return network

#function to get threshold values
def two_tail_thresh(graphs, threshold):
    '''
    Gives a set of thresholds to be used for weights of each specifc edge based off the distribution of edge wieghts in the set of graphs in.

    Param
    -----
    graphs: [networkx_graphs]
        List of networkx object graphs.
    threshold: int (0-50)
        the percentage of values to keep at each tail. (i.e. 15% means taking the top and bottom 15%)
    
    Returns
    -------
    thresholds: dict({node1|node2:(min_bound, max_bound)})
    '''

    values = {}
    for i, j in list(itertools.combinations(FEATURE_DICT.keys(), 2)):
        n1 = FEATURE_DICT[i]
        n2 = FEATURE_DICT[j]
        k = str(n1 + '|' + n2)
        values[k] = []

        for g in graphs:
            values[k].append(g[n1][n2]['Weight'])

    thresholds = {}
    for edge in values.keys():
        min_bound = np.percentile(values[edge], q=threshold)
        max_bound = np.percentile(values[edge], q=(100-threshold))
        thresholds[edge] = (min_bound, max_bound)
    
    return thresholds

#function to filter a graph given threshold values
def graph_filter(graph, thresholds):
    ''' 
    Filters edges of a graph based on the passed thresholds.

    Param
    -----
    graph: networkx object
        Graph to be filtered
    thresholds: dict({n1|n2:(min. max)})
        Dictionary holding ranges to be excluded for all edges in the graph.
    '''
    
    for i, j in list(itertools.combinations(FEATURE_DICT.keys(), 2)):
        n1 = FEATURE_DICT[i]
        n2 = FEATURE_DICT[j]
        k = str(n1 + '|' + n2)

        weight = graph.get_edge_data(n1, n2)['Weight']

        if thresholds[k][0] < weight < thresholds[k][1]:
            graph.remove_edge(n1, n2)

def draw_graph(graph, file_name):
    '''
    Saves the specified graph as a pdf to the specified file name.

    Param
    -----
    graph: networkx graph object
    file_name: str
        path to save the pdf to
    '''

    G = graph
    pos = nx.spring_layout(G)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos)

    # edges
    nx.draw_networkx_edges(G, pos)

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['Weight']:.2f}" for u, v, d in G.edges(data=True)})


    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(file_name)
    plt.close()

############################################################################################

PARAMETERS_FILE = sys.argv[1]
parameter_reader = parse_params(PARAMETERS_FILE)
try:
    DATA_DIR = parameter_reader['radiomics']['dir']
    CONNECT_DIR = parameter_reader['connectomics']['dir']

    FEATURE_DICT = parameter_reader['feature_dict']

    NUM_FEATURES = len(FEATURE_DICT)

    all_data = []
    for i in range(NUM_FEATURES):
        df = pd.read_csv(f'{DATA_DIR}/cleaned_results_pyrad_{str(i+1)}.csv')
        IMG_IDS = pd.DataFrame({'img_id':[x.split('/')[-1].split('.')[0] for x in list(df['Image'])]})
        df = df.iloc[:,3:]
        df = df.add_suffix(f'_radiomics_{i+1}')

        #think about other ways to normalize/clean the radiomics data here
        ####
        df = (df-df.mean())/df.std()
        df = df.fillna(0)
        ####


        df = pd.concat([IMG_IDS, df], axis=1)
        all_data.append(df)

    NUM_PT = len(IMG_IDS)

    print("Calculating weights...")
    #calculate distances and make graphs (these graphs are unfiltered at this point)
    connectomics_eucl = [process_image(img_id=i[0], distance_function=eucl_distance) for i in IMG_IDS.values]
    connectomics_cos = [process_image(img_id=i[0], distance_function=cosine_dist) for i in IMG_IDS.values]

    connectomics_eucl = dict(zip([i[0] for i in IMG_IDS.values], connectomics_eucl))
    connectomics_cos = dict(zip([i[0] for i in IMG_IDS.values], connectomics_cos))


    #get the thresholds and filter the graphs
    eucl_thresholds = two_tail_thresh(graphs=[g for g in connectomics_eucl.values()], threshold=parameter_reader["connectomics"]["two_tailed_filtering_level"])
    cos_thresholds = two_tail_thresh(graphs=[g for g in connectomics_cos.values()], threshold=parameter_reader["connectomics"]["two_tailed_filtering_level"])
    for g in tqdm(connectomics_eucl.values(), desc='eucl thresholds'): graph_filter(graph=g, thresholds=eucl_thresholds)
    for g in tqdm(connectomics_cos.values(), desc='cos thresholds'): graph_filter(graph=g, thresholds=cos_thresholds)


    #save pdfs of the graphs
    for n, g in tqdm(connectomics_eucl.items(), desc='drawing eucl graphs'): draw_graph(g, CONNECT_DIR+"/eucl_" + n + ".pdf")
    for n, g in tqdm(connectomics_cos.items(), desc='drawing cos graphs'): draw_graph(g, CONNECT_DIR+"/cos_" + n + ".pdf")  

    #calculate all of the graph statistics and export to a csv
    all_graph_stats = {}
    for target_graphs,graph_id in [(connectomics_eucl, 'euclidian_connectomics_'), (connectomics_cos, 'cosine_connectomics_')]:

        #setting up the labels of all the stats to be calculated
        all_data_labels = ['img_id', f'{graph_id}_average_clustering']
        tests_to_run_per_segment = {f'{graph_id}_betweenness_centrality': nx.betweenness_centrality, 
                                    f'{graph_id}_closeness_centrality': nx.closeness_centrality, 
                                    f'{graph_id}_eigenvector_centrality': nx.eigenvector_centrality}

        for l in tests_to_run_per_segment.keys():
            for i in range(NUM_FEATURES):
                all_data_labels.append(l + '_' + str(i+1))
        
        

        #setting up the dictionary that will hold the statsitics
        curr_graph_stats = {}
        for x in all_data_labels:
            curr_graph_stats[x] = []

        curr_graph_stats = pd.DataFrame(curr_graph_stats)
        

        #calculate the statistics for each graph and append to the graph_stats DF 
        for p,g in target_graphs.items():

            curr_stats = {'img_id':p, 
                        f'{graph_id}_average_clustering': [nx.average_clustering(g)]
                        }
            
            for l in tests_to_run_per_segment.keys():
                for i in range(NUM_FEATURES):
                    dlabel = (l + '_' + str(i+1))
                    curr_stats[dlabel] = [tests_to_run_per_segment[l](g)[FEATURE_DICT[str(i+1)]]]
            
            curr_graph_stats = pd.concat([curr_graph_stats, pd.DataFrame(curr_stats)])
        
        #copy stats for current graph to all_graphs dictionary
        for x,y in curr_graph_stats.items():
            all_graph_stats[x] = y


    all_graph_stats = pd.DataFrame(all_graph_stats)

    radio_connect_master_data = pd.merge(all_data.pop(), all_data.pop(), on='img_id')
    while len(all_data) > 0:
        radio_connect_master_data = pd.merge(radio_connect_master_data, all_data.pop(), on='img_id')

    radio_connect_master_data = pd.merge(radio_connect_master_data, all_graph_stats, on='img_id')
    radio_connect_master_data.to_csv(CONNECT_DIR+'/all_radio_connect.csv')

except Exception:
    print('Error: Could not complete connectomics pipeline. See traceback:')
    print(traceback.format_exc())