"""
A collection of useful functions!
"""
import os
import networkx as nx
import community
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-poster')
from multiprocessing import Pool
import itertools
import powerlaw
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


## GLOBAL VARIABLES THAT SHOULD BE CONSISTENT
DATA_DIR = '../../data/data_schoolofinf'
INST = [
    "institute for adaptive and neural computation",
    "institute for computing systems architecture",
    "institute of language cognition and computation",
    "institute of perception action and behaviour",
    "laboratory for foundations of computer science",
    "centre for intelligent systems and their applications",
    "neuroinformatics dtc"
]
NUM_INST = 10


def get_institute():
    return pkl.load(open(os.path.join(DATA_DIR, 'institutes.pkl'), 'rb'))

def get_lookup_pub(top=1997, bottom=2017):
    df = pd.read_pickle(os.path.join(DATA_DIR,'lookup_pub.pkl'))
    df.drop(df[(df.year <top) | (df.year>bottom)].index,inplace=True)
    return df

def get_lookup_poinf():
    return pd.read_pickle(os.path.join(DATA_DIR, 'lookup_poinf.pkl'))


def prepare_toks(top=1997, bottom=2017, with_pdf2txt=False):
    """Constraint the tokens tokens with the [top,bottom]
    
    Set with_pdf2txt to true if pdf tokens should be included; else ignore.
    """
    if with_pdf2txt:
        df = pd.read_pickle(os.path.join(DATA_DIR, 'toks', 'toks.combined.pkl'))
    else:
        df = pd.read_pickle(os.path.join(DATA_DIR, 'toks', 'toks.metadata.pkl'))
    
    # remove tokens outside of period limit:
    df.drop(
        df[(df.year < top) | (df.year > bottom)].index,
        inplace=True)
    if with_pdf2txt:
        df['toks_pdf2txt'] = df.toks_pdf2txt.apply(lambda x: [] if not len(x) else x)
    df['toks_metada'] = df.toks_metada.apply(lambda x: [] if not len(x) else x)
    
    return df


def get_poinf_pub_mapping():
    """
    poinf2pub mapping dataframe:
        index : id of researchers informatics
        pub_ids: (type:set) of publication ids that researcher participated in
    if poinf_id is not in the index, no publications 
        (using the native dataset, amounts to 64 individuals)
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'poinf2pub.df.pkl'))
    


def avg_degree_dist(degree_seq):
    n = np.sum(degree_seq)
    expected = 0
    for i, d in enumerate(degree_seq):
        expected += float(i) * float(d) / n
    # print(expected)
    return expected


def power_law_fit(degree_seq):

    fit = powerlaw.Fit(degree_seq, discrete=True, xmin=0.)  # Fit the data
    fig = fit.plot_ccdf(color='b', linewidth=2,
                        label='Empirical ccdf')  # plot the data
    fit.power_law.plot_ccdf(linestyle='--', color='b', linewidth=2,
                            label='Power Law Fit', ax=fig)  # plot the data
    alpha = fit.power_law.alpha

    fig.set_ylabel(r'$F(d) =P(D_{v} \geq d)$')
    fig.set_xlabel('d')
    fig.legend(loc='best')
    #     fig.set_title(r'Power law fit: $F(d) = (\frac{d}{d_{min}})^{-(\alpha-1)}$')
    fig.set_title(r'Power law fit ($\alpha=${:.2f})'.format(alpha))
    return fig


def degree_dist(G, show=True):
    """
    given a graph G, generate a degree histogram:
    """
    degree_seq = nx.degree_histogram(G)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    # n = len(G)
    # k = G.size()
    ax.hist(degree_seq)
    ax.set_xlabel('degree')
    ax.set_ylabel('count')
    ax.set_title('Average degree = {}'.format(avg_degree_dist(degree_seq)))
    plt.show()
    return ax, degree_seq


def generateGCC(G):
    """
    returns a list of connected components, sorted in reverse order - from largest to smallest
    Also print the size of each component wrt to the graph G
    """
    total_nodes = len(G)
    gccs = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
    percentage = []
    for i, g in enumerate(gccs, 1):
        percent = float(len(g)) / total_nodes
        percentage.append(percent)
        print('component {}: {:.2%}'.format(i, percent))
    return gccs, percentage


def clustering_coeff(G):
    c_coeff = nx.clustering(G)
    avg_c = np.mean(list(c_coeff.values()))
    return c_coeff, avg_c


def centrality_measure(G):
    nodes_centrallity = nx.degree_centrality(G)
    return nodes_centrallity


def partitions(nodes, n):
    "Partitions the nodes into n subsets"
    nodes_iter = iter(nodes)
    while True:
        partition = tuple(itertools.islice(nodes_iter, n))
        if not partition:
            return
        yield partition


def btwn_pool(G_tuple):
    return nx.betweenness_centrality_source(*G_tuple)


# To begin the parallel computation, we initialize a Pool object with the
# number of available processors on our hardware. We then partition the
# network based on the size of the Pool object (the size is equal to the
# number of available processors).
def between_parallel(G, processes=None):
    p = Pool(processes=processes)
    part_generator = 4 * len(p._pool)
    node_partitions = list(partitions(G.nodes(), int(len(G) / part_generator)))
    num_partitions = len(node_partitions)

    # Next, we pass each processor a copy of the entire network and
    # compute #the betweenness centrality for each vertex assigned to the
    # processor.

    bet_map = p.map(btwn_pool,
                    zip([G] * num_partitions, [True] * num_partitions,
                        [None] * num_partitions, node_partitions))

    # Finally, we collect the betweenness centrality calculations from each
    # pool and aggregate them together to compute the overall betweenness
    # centrality score for each vertex in the network.

    bt_c = bet_map[0]
    for bt in bet_map[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c


def adj_mat_to_graph(matrix, order, weighted=False):
    """Convert adjacency matrix to edgelist for use in nx

    Args:
        matrix: adjacency matrix
        order: The order of appearance of nodes
        weighted: if matrix is binary - simple graph/network or weights
    """
    _, h = np.shape(matrix)
    diag = np.diagonal(matrix)
    assert np.allclose(matrix.T, matrix, atol=1e-8), "matrix is NOT symmetric!"
    assert np.sum(diag) == 0, "non-zero diagonal matrix detected!"

    # only consider the upper triangle; above the diagonal zero
    matrix = np.triu(matrix)
    assert h == len(order),\
        "Number of individuals in matrix and order does not match; got: matrix = {}, order = {}".format(
            h, len(order))

    g = nx.Graph()

    for i, node_1 in enumerate(order):
        for w in range(h):
            # Only add edges that exists:
            if matrix[i][w] > 0:
                node_2 = order[w]
                assert node_2 != node_1, "self-loop detected"
                g.add_edge(node_1, node_2, weight=matrix[i][w])
    return g


def create_adj_mat(g, order, draw=False, use_order=True, weighted=False):
    """
    Create the adjacenecy in a given order (list of node id)
    using the given graph g
    """
    if not use_order:
        nodes_in_g = list(g.nodes)
        order = [a for a in order if a in nodes_in_g]

    adj_mat = np.zeros([len(order), len(order)])
    for i, node in enumerate(order):
        try:
            neighbours = [n for n in g[node]]
            for neighbour in neighbours:
                idx = order.index(neighbour)
                if weighted:
                    adj_mat[i][idx] = adj_mat[i][idx] = g[node][neighbour]['weight']
                else:
                    adj_mat[i][idx] = 1
        except KeyError:
            # This happens when the individuals in the ORDER is not
            # in the graph g
            pass

    fig = None
    # Draw the graph:
    if draw:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111)
        sns.heatmap(
            adj_mat,
            square=True,
            yticklabels=order,
            xticklabels="",
            ax=ax,
            cbar=False)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(5)

    return adj_mat, fig, order

def convert_to_dict(tuple_of_partition):
    ret = {}
    idx=1
    for community in tuple_of_partition:
        for _id in community:
            ret[_id] = idx
        idx+=1
    return ret


def get_best_gnalgo(g):
    comm = nx.community.girvan_newman(g)
    states=[]
    scores = []
    best_score = 0    
    best_partition = None
    # starting state:
    _g = g
    org_g = g
    for partition in comm:
        _partition = convert_to_dict(partition)
        _score = community.modularity(_partition, _g) 
        # compute score based on previous induced graph
        states.append(_partition)
        scores.append(_score)
        
        if _score > best_score:
            best_score = _score
            best_partition = _partition
            logging.info('best_score: {:.3f}'.format(best_score))
        
    return best_partition, best_score

def inverse_partition(partition_dict):
    ret = {}
    for i in sorted(list(set(partition_dict.values()))):
        ret[i] = [a for (a, b) in partition_dict.items() if b == i]
    return ret


def create_community_graph(partition, g):
    graphs = []
    lookup = inverse_partition(partition)
    for comm_id in sorted(list(set(partition.values()))):
        graphs.append(g.subgraph(lookup[comm_id]))
    return nx.compose_all(graphs)