from ldamodel_trainer import find_topics, write_scores
import logging
import os
import warnings
import pandas as pd
from gensim.corpora import Dictionary
warnings.filterwarnings('ignore')
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
  DATA_DIR = '../../data/data_schoolofinf/'
  store_dir = os.path.join(os.getcwd(), 'tmfull_meta')
  # Parameters FOR LDA:
  num_topics = [10, 15, 20, 25, 30, 35, 40, 50]

  # Import the dataset:
  df_combined_toks = pd.read_pickle(
      os.path.join(DATA_DIR, 'toks', 'toks.combined.pkl'))
  # filter publications
  df_combined_toks = df_combined_toks.drop(df_combined_toks[(
      df_combined_toks.year < 1997) | (df_combined_toks.year > 2017)].index)
  # only use the metadata
  df_combined_toks['toks_metada'] = df_combined_toks.toks_metada.apply(
      lambda x: [] if not len(x) else x)
  # load dictionary required to convert tokens to idx
  dict_meta = Dictionary.load(os.path.join(
      DATA_DIR, 'corpora', 'dictionary.meta'))
  df_metadata = df_combined_toks[['year', 'toks_metada']]
  # Create a bow tagging for each publication:
  df_metadata['bow'] = df_metadata['toks_metada'].apply(dict_meta.doc2bow)

  # Generate a corpus based on the tokens, which we will be using later
  corpus_meta = df_metadata.bow.tolist()

  coherence_scores = find_topics(
      num_topics, corpus_meta, texts=df_metadata.toks_metada.tolist(),
      dictionary=dict_meta, store_dir=store_dir)
  print(coherence_scores)
  write_scores(coherence_scores, store_dir)
