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

from helper_embedding import *

if __name__ == "__main__":
  lookup_combined_toks = prepare_toks(top=2012, bottom=2017)
  print(len(lookup_combined_toks))
  df_pubmapping = get_poinf_pub_mapping()

  # Load topic models:
  tmrest_meta = models.LdaModel.load('../../topicModel/src/tmrest_meta/res/best_ldamodel')
  tmrest_meta.minimum_probability = 0.01
  dict_tmrest_meta = Dictionary.load(
      os.path.join(DATA_DIR, 'corpora', 'dictionary.less.meta'))

  # load collaboration network:
  infnet6yr = pd.read_csv(
      os.path.join(DATA_DIR, 'poinf_collabgraph_2012-2017.txt'),
      sep='\n',
      names=['id'])

  # Combine all tokens:
  topicnet6yr = infnet6yr.set_index('id').join(
      df_pubmapping.set_index('id'), how='inner')
  topicnet6yr['toks'] = topicnet6yr['pub_ids'].apply(
      lambda a: gen_toks(a, lookup_combined_toks))
  topicnet6yr['tm6yr_probs'] = topicnet6yr['toks'].apply(
      lambda x: tmrest_meta.get_document_topics(
          dict_tmrest_meta.doc2bow(x)) if len(x) else None
  )

  # Calculate cosine similarity
  cosim = compare_researchers(
      topicnet6yr.tm6yr_probs.tolist(), tmrest_meta.num_topics)

  # load collaboration graph adjacency matrix
  ground_truth_adj_mat = np.load(os.path.join(
      DATA_DIR, 'mat', 'infnet6yrs-adj-mat.pkl'))

  # Find optimum threshold
  thresholds, edges, distances, closest_edges, best_threshold, lowest_avg_distance, best_j_dist_epoch, best_threshold_j_dist = \
      find_best_threshold(ground_truth_adj_mat, cosim,
                          num_iter=10000000, step_size=0.001)

  # Plot graphs:
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111)

  l0 = ax.plot(thresholds, distances, 'b', label='Avg Jaccard Distance')
  ax.set_xlabel('Thresholds')
  ax.set_ylabel('Average Jaccard Distance')

  ax2 = ax.twinx()
  l1 = ax2.plot(thresholds, edges, 'r-.',
                label='Num edges in topic-collab net')
  ax2.set_ylabel('Total Number of Edges')

  l2 = ax2.plot(
      np.linspace(0, 1., 100),
      np.repeat(np.sum(ground_truth_adj_mat) / 2, 100),
      'g:',
      label='Num edges in collab net')

  ls1 = ax.scatter(best_threshold_j_dist, lowest_avg_distance, facecolors='c',
                   edgecolors='c', alpha=.2, label='Lowest avg jaccard dist epoch')
  ls2 = ax2.scatter(best_threshold, closest_edges, facecolors='m',
                    edgecolors='m', alpha=.2, label='Closest epoch to ground-truth')

  fig.legend(loc='upper center')
  plt.tight_layout()
  fig.savefig('IMG/topicnet6yr_thresholding.png',
              format='png', bbox_inches='tight')
