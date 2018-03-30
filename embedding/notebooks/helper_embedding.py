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


def prepare_toks(top=1997, bottom=2017):
  """Constraint the tokens tokens with the [top,bottom]
  """
  # Tokens from collection
  lookup_combined_toks = pd.read_pickle(
      os.path.join(DATA_DIR, 'toks', 'toks.combined.pkl'))
  lookup_combined_toks.drop(
      lookup_combined_toks[(lookup_combined_toks.year < top)
                           | (lookup_combined_toks.year > bottom)].index,
      inplace=True)
  lookup_combined_toks[
      'toks_pdf2txt'] = lookup_combined_toks.toks_pdf2txt.apply(
          lambda x: [] if not len(x) else x)
  lookup_combined_toks[
      'toks_metada'] = lookup_combined_toks.toks_metada.apply(
          lambda x: [] if not len(x) else x)

  return lookup_combined_toks


def get_poinf_pub_mapping():
  df_pubmapping = pd.read_pickle(
      os.path.join(DATA_DIR, 'poinf_to_pub_mapping.pkl'))
  return df_pubmapping


def _to_list(l, target_length):
  """
  List of tuples...
  """
  out = np.zeros(target_length, dtype=np.float32)
  for (i, v) in l:
    out[i] = v
  return out

# Function used to generate a list of tokens based on the publications of an individual (pub_ids)


def gen_toks(pub_ids, df_toks):
  """
  Take in a set of pub_ids and concatenate all the tokens together
  """
  _indices = set(df_toks.index)
  pub_ids = pub_ids.intersection(_indices)
  _df = df_toks.loc[list(pub_ids)]  # slice the tokens to get those
  out = []
  for a in _df.toks_metada.tolist():  # convert reduce the list of list
    out += a
  return out


def _cosine(a, b):
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
      sim_matrix[i][j] = 1. - dist  # cosine(a,b) outputs the distance
    sim_matrix[i][i] = 0.
  return sim_matrix


def jaccard_dist(x_true, x, theta=None, binary=True):
  """
  Given vectors x_true and x of the same length, calculate the
  jaccard distance between x_true and x.
  Theta is represents the threshold parameter to filter x by.
  If given, any values in x < theta will be set to 0
  """
  x = np.array(x)
  if binary:
    if theta:
      idx = x < theta
      x[idx] = False
      x[~idx] = True
#             print(x)
      # np.sum(x) = number of edges (1s)
      return jaccard(x_true, x), np.sum(x, dtype=int)
    else:
      return jaccard(x_true, x)
  else:
    raise("not implemented")


def binom_choose(n, k):
  """Implements n choose k
  """
  assert k <= n, "k must be less than or equal to n"
  import math
  denom = math.factorial(k) * math.factorial(n - k)
  return math.factorial(n) / denom


def set_edges(matrix, threshold, binary=True):
  _matrix = np.zeros_like(matrix, dtype=np.int32)

  w, h = np.shape(matrix)
  logging.info('dimension: {}, {}'.format(w,h))
  for i in range(w):
    for j in range(h):
      if binary:
        _matrix[i][j] = 1 if matrix[i][j] > threshold else 0.
      else:
        _matrix[i][j] = matrix[i][j] if matrix[i][j] > threshold else 0.
  return _matrix


def find_best_threshold(ground_truth_adj_mat,
                        sim_matrix,
                        start_threshold=0.005,
                        step_size=0.005,
                        num_iter=100,
                        verbose=True):
  """
  An iterative approach to find the best threshold such that
  when sim_matrix is under binary threshold, it is the most similar to graound_truth_adj_mat
  that is binary.

  The condition used can be changed.git pu
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
  ground_truth_sum = np.sum(ground_truth_adj_mat)
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
      j_dist, num_edge = jaccard_dist(x_true, x, theta=threshold, binary=True)
      distances[i] = j_dist
      num_edges += num_edge
      i += 1

    # calculate the mean for a given threshold
    average_dist = np.mean(distances)
    _NUM_EDGES.append(num_edges)
    _THRESHOLDS.append(threshold)
    _AVG_DISTANCES.append(average_dist)

    if verbose:
      logging.info(("epoch {}: threshold: {:.3f} avg_dist: {:.3f} num_edges: {}".
                    format(epoch, threshold, average_dist, num_edges)))

    # check if the best epoch is seen
    #         if (average_dist < lowest_avg_distance):
    #             best_epoch = epoch
    #             lowest_avg_distance = average_dist
    #             best_threshold = threshold

    if (num_edges >= ground_truth_sum):
      # if (num_edges < ground_truth_sum):
      best_epoch = epoch
      best_threshold = threshold
      maximum_edges = num_edges

    if average_dist < lowest_avg_distance:
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
                   j_dist_best_threshold, lowest_j_distance, ground_truth_adj_mat ):
    # Plot graphs:
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111)

  ax.plot(thresholds, distances, 'b', label='Avg Jaccard Distance')
  ax.set_xlabel('Thresholds')
  ax.set_ylabel('Average Jaccard Distance')

  ax2 = ax.twinx()
  ax2.plot(thresholds, edges, 'r-.',
                label='Num edges in topic-collab net')
  ax2.set_ylabel('Total Number of Edges')

  ax2.plot(
      np.linspace(0, 1., 100),
      np.repeat(np.sum(ground_truth_adj_mat) / 2, 100),
      'g:',
      label='Num edges in collab net')

  ax.scatter(j_dist_best_threshold, lowest_j_distance, facecolors='c',
                   edgecolors='c', alpha=.2, label='Lowest avg jaccard dist epoch')
  ax2.scatter(best_threshold, lowest_edges, facecolors='m',
                    edgecolors='m', alpha=.2, label='Closest epoch to ground-truth')

  fig.legend(loc='upper center')
  plt.tight_layout()
  return fig
