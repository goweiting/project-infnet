from ..ldamodel_trainer import find_topics
import csv
import logging
import os
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')
import pickle as pkl
import pandas as pd
from numpy.random import RandomState
from gensim import models
from gensim.corpora import Dictionary

passes = 100  # total number of times the corpus is seen
iterations = 1000  # how many times each document is seen
chunksize = 2000  # how many documents each mini-batch
update_every = 2000  # Batch learning
eval_every = 2000
rng = RandomState(93748573)


if __name__ == '__main__':
  DATA_DIR = '../../../data/data_schoolofinf/'
  # Parameters FOR LDA:
  num_topics = [10, 15, 20, 25, 30, 35, 40, 50]

  # Import the dataset:
  df_combined_toks = pd.read_pickle(
      os.path.join(DATA_DIR, 'toks', 'toks.combined.pkl'))
  df_combined_toks = df_combined_toks.drop(df_combined_toks[(
      df_combined_toks.year < 1997) | (df_combined_toks.year > 2017)].index)
  df_combined_toks['toks_pdf2txt'] = df_combined_toks.toks_pdf2txt.apply(
      lambda x: [] if not len(x) else x)
  df_combined_toks['toks_metada'] = df_combined_toks.toks_metada.apply(
      lambda x: [] if not len(x) else x)

  dict_all = Dictionary.load(os.path.join(
      DATA_DIR, 'corpora', 'dictionary.all'))
  df_all = df_combined_toks.copy()
  df_all['concat_toks'] = df_combined_toks.apply(
      lambda row: row.toks_metada + row.toks_pdf2txt, axis=1)
  # Create a bow tagging for each publication:
  df_all['bow'] = df_all['concat_toks'].apply(dict_all.doc2bow)

  # Generate a corpus based on the tokens, which we will be using later
  corpus_all = df_all.bow.tolist()

  # The function
  coherence_scores = find_topics(
      num_topics, corpus_all, texts=df_all.concat_toks.tolist(),
      dictionary=dict_all, store_dir=os.getcwd() )
  # pkl.dump(coherence_scores, open('./coherencescores.pkl', 'wb'))
  with open(store_dir + 'scores.csv', 'w') as f:
    fieldnames = ['num_topic', 'c_v', 'u_mass']
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    for num_topic in coherence_scores.keys():
      writer.writerow({'num_topics': num_topic,
                       'c_v': coherence_scores[num_topic]['c_v'],
                       'u_mass': coherence_scores[num_topic]['u_mass']})
  print(coherence_scores)
