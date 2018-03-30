import os
import pickle as pkl
import networkx as nx
# random_state for networkX only for python3.6
from numpy.random import RandomState
rng = RandomState(787351)
import matplotlib.pyplot as plt
plt.style.use(['seaborn-poster'])
import pandas as pd
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Global vars
DATA_DIR = '../../data/data_schoolofinf'
INSTITUTES = pkl.load(open(os.path.join(DATA_DIR, 'institutes.pkl'), 'rb'))

# DEFINE COLORS
# inst_by_color = {
#     0: 'xkcd:silver',
#     1: 'xkcd:cyan',
#     2: 'xkcd:magenta',
#     3: 'xkcd:indigo',
#     4: 'xkcd:red',
#     5: 'xkcd:lime',
#     6: 'xkcd:goldenrod',
#     7: 'xkcd:azure',
#     8: 'xkcd:lavender',
#     9: 'xkcd:grey green',
#     10: 'xkcd:coral',
#     11: 'xkcd:khaki',
#     'others': 'xkcd:claret'
# }

inst_by_color = {
    0: '#000000',
    1: '#0000ff',
    2: '#00ffff',
    3: '#00cc00',
    4: '#ff9900',
    5: '#ff0000',
    6: '#F20BCE',
    7: '#999966',
    8: '#ccffff',
    9: '#ffffb3',
    10: '#e6e6ff',
    11: '#e6f2ff',
    'others': '#808080'
}


def color_by_inst(g, lookup_poinf):
    # light up the nodes based on the institutes they belong to:
    node_color = []
    for node in g:
        node_color.append(inst_by_color[int(
            lookup_poinf.institute_class.loc[[str(node)]])])
    return node_color


def add_inst_labels(ax, with_legend=True):
    # Append legend into the axis
    for label in list(INSTITUTES.values()):
        ax.scatter(
            [0], [0],
            color=inst_by_color[label],
            label=[
                name for (name, _k) in list(INSTITUTES.items()) if _k == label
            ][0])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.scatter([0], [0], color='white', s=100, edgecolors='none')
    return ax



def draw_default_layout(g,
                        lookup_poinf,
                        file_prefix,
                        with_weight=False,
                        scale=2,
                        SAVE_GRAPHS=False):
    """
    function to create the same layout for report
    """
    logging.info('SAVE_GRAPHS: {}'.format(SAVE_GRAPHS))

    # Create dictionary where each key is the institute and value is the
    # id of nodes in that institute; basically separated the individuals into
    # institutes (for shell layout)

    _nlist = get_default_nlist(as_dict=False)
    nlist_merged = get_default_nlist(as_dict=True)

    # Cross institute collaboration
    edges = []
    for (a, b) in g.edges:
        c_a = int(lookup_poinf.institute_class.loc[[str(a)]])
        c_b = int(lookup_poinf.institute_class.loc[[str(b)]])
        if c_a != c_b:
            if (b, a) not in edges:
                edges.append((a, b))

    in_school = [i for i in range(1, 8)]

    # DRAW
    fig = plt.figure(figsize=(10, 10))
    # plot the connection between communities (excluding within the same community) in the first 3,3:
    # ax = plt.subplot2grid((6,3), (0,0), colspan=3, rowspan=3)
    ax = fig.add_subplot(111)
    ax.axis('off')

    edgewidth = 1
    if with_weight:
        edgewidth = [
            d['weight'] * float(scale) for (u, v, d) in g.edges(data=True)
        ]
    nx.draw_networkx(
        g,
        pos=nx.shell_layout(g, _nlist),
        with_labels=False,
        ax=ax,
        edge_color='#999966',
        node_size=60,
        node_color=color_by_inst(g),
        edgelist=edges,  # <- Cross collaborations!
        width=edgewidth)

    if SAVE_GRAPHS:
        plt.savefig(
            "IMG/{}_shell_BTW_INST.png".format(file_prefix),
            format='png',
            bbox_inches="tight",
            transparent=True)

    # Draw each institute using circular layout:
    #     row_idx = 3
    #     col_idx = 0
    for i in in_school:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        #     ax = plt.subplot2grid((6,3),(row_idx,col_idx))
        ax.axis('off')
        _g = g.subgraph(nlist_merged[i])

        edgewidth = 1
        if with_weight:
            edgewidth = [
                d['weight'] * float(scale)
                for (u, v, d) in _g.edges(data=True)
            ]
        nx.draw_networkx(
            _g,
            pos=nx.circular_layout(_g),
            with_labels=False,
            ax=ax,
            edge_color='#999966',
            node_size=60,
            node_color=color_by_inst(_g),
            width=edgewidth)
        #         col_idx += 1
        #         col_idx = col_idx % 3
        #         if col_idx == 0:
        #             row_idx += 1
        #     inst = [name for (name, _k) in list(institutes.items()) if _k == k][0]
        #     title = "{: <5}".format(inst)
        #     ax.set_title(title)

        if SAVE_GRAPHS:
            plt.savefig(
                "IMG/{}_circular_inst_{}.png".format(file_prefix, i),
                format='png',
                bbox_inches="tight",
                transparent=True)
    # plt.subplots_adjust()
    # plt.tight_layout()


def get_default_nlist(lookup_poinf, as_dict=False):
    """
    Group individuals according to their institute
    Returns a dictionary with key = 1 to 7, representing the institutes
    An additional 'others' group for all the UNKNOWN and instittues not in the school.
    """

    nlist = {}
    for k, gb in lookup_poinf.groupby('institute_class'):
        nlist[k] = gb.index.tolist()
    # Put Non-official classes as one list - 'others':
    in_school = [i for i in range(1, 8)]
    nlist_merged = {'others': []}

    for i in list(nlist.keys()):
        if i in in_school:
            nlist_merged[i] = nlist[i]
        else:
            nlist_merged['others'].extend(nlist[i])

    if as_dict:
        return nlist_merged
    else:
        # return as a list
        _nlist = [nlist_merged[a] for a in in_school]
        _nlist.append(nlist_merged['others'])
        return _nlist


def draw_circular_layout(g, with_weight=False, scale=2, file_prefix=None, SAVE_GRAPHS=False):
    # generate pos

    logging.info('SAVE_GRAPHS: {}'.format(SAVE_GRAPHS))
    nlist = get_default_nlist()
    _nlist = []
    for x in nlist:
        _nlist.extend(x)
    pos = nx.circular_layout(_nlist, scale=4)


    # visualise:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')

    edgewidth = 1
    if with_weight:
        edgewidth = [
            d['weight'] * float(scale) for (u, v, d) in g.edges(data=True)
        ]

    nx.draw_networkx(
        g,
        pos=pos,
        with_labels=False,
        ax=ax,
        node_size=30,
        edge_color='#999966',
        node_color=color_by_inst(g),
        width=edgewidth)

    if SAVE_GRAPHS:
        # ax.set_title('Informatics Collaboration Network from 1997-2017')
        #     ax = add_inst_labels(ax) #You can include the label by uncommenting `ax = add_inst_label(ax)`
        #plt.savefig("IMG/infnet20yr_spring.pdf", format='pdf', bbox_inches="tight")
        plt.savefig(
            "IMG/{}_circular.png".format(file_prefix),
            format='png',
            bbox_inches="tight",
            transparent=True)
