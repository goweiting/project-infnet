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
  lookup_combined_toks = prepare_toks()
  df_pubmapping = get_poinf_pub_mapping()

  # Load topic models:
  tmdblp = models.LdaModel.load('../../topicModel/src/tmdblp/tm_dblp')
  tmdblp.minimum_probability = 0.01
  dict_tmdblp = Dictionary.load(
      '../../data/data_dblp/corpora/dictionary.dblp.1997-2017')

  # load collaboration network:
  infnetref20yr = pd.read_csv(
      os.path.join(DATA_DIR, 'poinf_collabgraph_1997-2017.txt'),
      sep='\n',
      names=['id'])
  # load collaboration graph adjacency matrix
  ground_truth_adj_mat = np.load(os.path.join(
      DATA_DIR, 'mat', 'infnet20yrs-adj-mat.pkl'))

  # Combine all tokens:
  topicnetref20yr = infnetref20yr.join(
      df_pubmapping.set_index('id'), how='inner', on='id')
  topicnetref20yr['toks'] = topicnetref20yr['pub_ids'].apply(
      lambda a: gen_toks(a, lookup_combined_toks))
  topicnetref20yr['tm20yr_probs'] = topicnetref20yr['toks'].apply(
      lambda x: tmdblp.get_document_topics(
          dict_tmdblp.doc2bow(x)) if len(x) else None
  )

  # Calculate cosine similarity
  cosim = compare_researchers(
      topicnetref20yr.tm20yr_probs.tolist(), tmdblp.num_topics)

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
  fig.savefig('IMG/topicnetref20yr_thresholding2.png',
              format='png', bbox_inches='tight')
  # threshold: 0.845 dist 0.794877 (689, 0.694) edges 946/940.0 (840)
