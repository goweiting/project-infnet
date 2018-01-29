"""
A collection of useful functions!
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
import powerlaw

# DEFINE COLORS
inst_by_color = {
    0: 'xkcd:silver',
    1: 'xkcd:cyan',
    2: 'xkcd:magenta',
    3: 'xkcd:indigo',
    4: 'xkcd:red',
    5: 'xkcd:lime',
    6: 'xkcd:goldenrod',
    7: 'xkcd:azure',
    8: 'xkcd:lavender',
    9: 'xkcd:grey green',
    10: 'xkcd:coral',
    11: 'xkcd:khaki',
    'others': 'xkcd:claret'
}

def avg_degree_dist(degree_seq):
    n = np.sum(degree_seq)
    expected = 0
    for i, d in enumerate(degree_seq):
        expected += float(i) * float(d) / n
    # print(expected)
    return expected

def power_law_fit(degree_seq):

    fit = powerlaw.Fit(degree_seq, discrete=True, xmin=0.) # Fit the data
    fig = fit.plot_ccdf(color='b', linewidth=2, label='Empirical ccdf') # plot the data
    fit.power_law.plot_ccdf(linestyle='--', color='b', linewidth=2, label='Power Law Fit', ax=fig) #plot the data
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

    #Next, we pass each processor a copy of the entire network and
    #compute #the betweenness centrality for each vertex assigned to the
    #processor.

    bet_map = p.map(btwn_pool,
                    zip([G] * num_partitions, [True] * num_partitions,
                        [None] * num_partitions, node_partitions))

    #Finally, we collect the betweenness centrality calculations from each
    #pool and aggregate them together to compute the overall betweenness
    #centrality score for each vertex in the network.

    bt_c = bet_map[0]
    for bt in bet_map[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c