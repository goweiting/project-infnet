#!/bin/python

"""
python script for generating topicnetref20yr
this collaboration-topic network uses:
1) collaboration network: infnet20yr
2) topic model: tm20yr
"""

# Import modules:
import pickle as pkl
import pandas as pd
import gensim.models as models
import gensim.similarities as sim
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.manifold import MDS
from scipy.spatial.distance import cosine, jaccard
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['seaborn-poster'])

import logging
import os
import warnings
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
warnings.filterwarnings('ignore')

DATA_DIR = '../../data/data_schoolofinf/'


def _to_list(l, target_length):
  """
  List of tuples...
  """
  out = np.zeros(target_length, dtype=np.float32)
  for (i, v) in l:
    out[i] = v
  return out


def gen_toks(pub_ids, df_toks):
  """
  Function used to generate a list of tokens based on the publications of an individual (pub_ids)
  Take in a set of pub_ids and concatenate all the tokens together

  Arguments:
    pub_ids : list of publication ids
    df_toks : pandas dataframe with all the publications under consideration
  """
  # Only use publications that are in df_toks
  _indices = set(df_toks.index)
  pub_ids = pub_ids.intersection(_indices)
  # slice the tokens to get publications in pub_ids
  _df = df_toks.loc[list(pub_ids)]
  out = []
  for a in _df.toks_metada.tolist():  # convert reduce the list of list
    out += a
  return out


def _cosine(a, b):
  #  my implementation of cosine similarity
  a_norm = np.linalg.norm(a)
  b_norm = np.linalg.norm(b)
  return np.dot(a, b) / (a_norm * b_norm)


def compare_researchers(list_of_probs, n_topics):
  """Takes in a list of probabilities inferrred (using LdaModel)
  and calculate cosine similarity between each (n 2) pairs of researchers

  Output:
    [N N] matrix where N = len(list_of_probs)
  """
  num_individuals = len(list_of_probs)
  sim_matrix = np.zeros([num_individuals, num_individuals])

  for i in range(num_individuals):
    a = _to_list(list_of_probs[i], n_topics)
    assert all(i >= 0 for i in a), "negative probabilities?"
    for j in range(num_individuals):
      b = _to_list(list_of_probs[j], n_topics)
      assert all(i >= 0 for i in b)
      dist = cosine(a, b)
      # assert dist <= 1. and dist >= 0., "negative distance {}?, a:{}, b:{}".format(dist, a,b)
      # cosine(a,b) outputs the distance (see scipy.spatial.distance)
      sim_matrix[i][j] = 1. - dist
    sim_matrix[i][i] = 0.
  return sim_matrix


def jaccard_dist(x_true, x, theta):
  """
  Given vectors x_true and x of the same length, calculate the
  jaccard distance between x_true and x.
  Theta is represents the threshold parameter to filter x by.
  If given, any values in x < theta will be set to 0
  """
  x = np.array(x)
  idx = x < theta

  x[idx] = False
  x[~idx] = True
  num_edges = np.sum(x, dtype=int)  # CountEdges
  j_dist = jaccard(x_true, x)  # calculates the local jaccard distance
  return j_dist, num_edges


def binom_choose(n, k):
  """Implements n choose k
  """
  assert k <= n, "k must be less than or equal to n"
  import math
  denom = math.factorial(k) * math.factorial(n - k)
  return math.factorial(n) / denom


def set_edges(matrix, threshold, binary=True):

  w, h = np.shape(matrix)
  logging.info('dimension: {}, {}'.format(w, h))
  if binary:
    _matrix = np.zeros_like(matrix, dtype=np.int32)
    _matrix[matrix > threshold] = 1
  else:
    _matrix = matrix
    _matrix[matrix <= threshold] = 0
#   for i in range(w):
#     for j in range(h):
#       if binary:
#         _matrix[i][j] = 1 if matrix[i][j] > threshold else 0.
#       else:
#         _matrix[i][j] = matrix[i][j] if matrix[i][j] > threshold else 0.
  return _matrix


def find_best_threshold(ground_truth_adj_mat,
                        sim_matrix,
                        binary_edges=True,
                        start_threshold=0.001,
                        step_size=0.005,
                        num_iter=100,
                        verbose=True):
  """An iterative approach to find the best threshold such that
  the number of edges in sim_matrix subjected to a threshold is most similar
  to the ground_truth_adj_mat

  The condition used can be changed.

  Args:
    ground_truth_adj_mat: the collaboration graph represent in adjacency matrix
    sim_matrix: the cosine similarity matrix derived from compare_researchers
    binary_edges: Set to False if the ground_truth_adj_mat is not binary (default: {True})
    start_threshold: Threshold to begin the iterative process with; every step increases the threshold
                     (default: {0.005})
    step_size: increment per iteration (default: {0.001})
    num_iter: total number of iteration (default: {100})
    verbose: Set to true to print the iteration progress (default: {True})
  """

  # Initialise the parameters
  end_conditions = True
  lowest_avg_distance = 1  # maximum possible
  nb_individuals = ground_truth_adj_mat.shape[0]
  assert nb_individuals == sim_matrix.shape[0],\
      "adjacency matrices and similarity matrix must have the same shape! adj mat: {} ; sim mat: {}".format(
      ground_truth_adj_mat.shape, sim_matrix.shape)

  epoch = 0
  maximum_edges = 0
  if binary_edges:
    ground_truth_sum = np.sum(ground_truth_adj_mat) // 2
  else:
    # get the minimum; larger than 0
    min_weight = np.min(ground_truth_adj_mat[ground_truth_adj_mat > 0])
    ground_truth_sum = np.sum(ground_truth_adj_mat >= min_weight) // 2
  logging.info('Number of ground_truth_edges: {}'.format(ground_truth_sum))
  logging.info('binary edges: {}'.format(binary_edges))
  threshold = start_threshold
  best_threshold = 0.
  best_epoch = 0
  best_epoch_j_dist = 0
  best_threshold_j_dist = 0.

  # stores the history of the progress:
  _NUM_EDGES = []
  _AVG_DISTANCES = []
  _THRESHOLDS = []

  while (end_conditions):

    i = 0
    num_edges = 0  # calculate the number of edges for this threshold.
    distances = np.zeros(nb_individuals)  # store the average distances

    for x_true, x in zip(ground_truth_adj_mat, sim_matrix):
      # calculate local statistics for each node
      # its convered in jaccard_dist because we calculate the jaccard distance
      # between each individual too.
      j_dist, num_edge = jaccard_dist(x_true, x, theta=threshold)
      distances[i] = j_dist
      num_edges += num_edge
      i += 1
    num_edges = num_edges // 2
    # calculate the mean for a given threshold
    average_dist = np.mean(distances)
    _NUM_EDGES.append(num_edges)
    _THRESHOLDS.append(threshold)
    _AVG_DISTANCES.append(average_dist)

    if verbose:
      logging.info(("epoch {}: threshold: {:.3f} avg_dist: {:.3f} num_edges: {}".
                    format(epoch, threshold, average_dist, num_edges)))

    if (num_edges >= ground_truth_sum):
      best_epoch = epoch
      best_threshold = threshold
      maximum_edges = num_edges

    if average_dist < lowest_avg_distance:
      # Additionally, record the local average jaccard distance
      best_epoch_j_dist = epoch
      best_threshold_j_dist = threshold
      lowest_avg_distance = average_dist

    # Next iteration
    epoch += 1
    threshold += step_size
    end_conditions = (epoch < num_iter) and (
        threshold <= 1.)  # condition to draw graph
    # end_conditions = (epoch < num_iter) and (num_edges >= ground_truth_sum) and (threshold <= 1) # condition

  print(('threshold: {:.3f} dist {:3f} ({}, {:.3f}) edges {}/{} ({})'.format(
      best_threshold, lowest_avg_distance, best_epoch_j_dist,
      best_threshold_j_dist, maximum_edges, ground_truth_sum, best_epoch)))

  return _THRESHOLDS, _NUM_EDGES, _AVG_DISTANCES,\
      maximum_edges, best_threshold, lowest_avg_distance,\
      best_epoch_j_dist, best_threshold_j_dist


def threshold_plot(thresholds, distances, edges, best_threshold, lowest_edges,
                   j_dist_best_threshold, lowest_j_distance,
                   ground_truth_num_edges, cosim):
    # Plot graphs:
  fig = plt.figure(figsize=(8, 9))
  ax = fig.add_subplot(211)

  ax.plot(
      thresholds, edges, 'g-', label='Num edges in topic-collab net')
  ax.scatter(
      best_threshold,
      lowest_edges,
      facecolors='m',
      edgecolors='m',
      alpha=.4,
      label='Best threshold (number of edges)')
  ax.plot(
      np.linspace(0, 1., 100),
      np.repeat(ground_truth_num_edges, 100),
      'r:',
      dashes=(5, 10), alpha=.5,
      label='Num edges in collab net')
  ax.set_yscale('log')
  ax.set_ylabel('Number of Edges')
  ax.yaxis.label.set_color('g')
  ax.xaxis.set_label_position('top')
  ax.set_xlabel('Threshold, $\epsilon$')
  ax.xaxis.tick_top()
  ax.set_xticklabels(thresholds, minor=True)

  ax2 = ax.twinx()
  ax2.plot(thresholds, distances, 'b-.', label='Avg Jaccard Distance')
  ax2.scatter(
      j_dist_best_threshold,
      lowest_j_distance,
      facecolors='c',
      edgecolors='c',
      alpha=.4,
      label='Best threshold (avg jaccard dist)')
  ax2.yaxis.label.set_color('b')
  ax2.set_ylabel('Average jaccard distance')

  ax3 = fig.add_subplot(212, sharex=ax2)
  dist = np.triu(cosim)
  dist = np.ravel(dist[dist > 0])
  sns.distplot(
      dist, hist=True, ax=ax3, label='Cosine similarity btw researchers',
      color="k")
  ax3.set_xlabel('Cosine Similarity')
  ax3.set_ylabel('Number of researchers')
  ax3.plot(
      np.repeat(best_threshold, 10),
      np.linspace(0, 6, num=10),
      'm--',
      dashes=(5, 10),
      alpha=.4)
  ax3.plot(
      np.repeat(j_dist_best_threshold, 10),
      np.linspace(0, 6, num=10),
      'c--',
      dashes=(5, 10),
      alpha=.4)

  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  h3, l3 = ax3.get_legend_handles_labels()

  lns = h1 + h2 + h3
  labels = [l.get_label() for l in lns]

  ax3.legend(lns, labels, loc=0)
  plt.tight_layout()
  plt.subplots_adjust(hspace=0)
  return fig
