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
  store_dir = os.path.join(os.getcwd(), 'tmrest')
  # Parameters FOR LDA:
  num_topics = [10, 15, 20, 25, 30, 35, 40, 50]

  # Import the dataset:
  df_combined_toks = pd.read_pickle(
      os.path.join(DATA_DIR, 'toks', 'toks.combined.pkl'))
  df_less_all = df_combined_toks.drop(
      df_combined_toks[(df_combined_toks.year < 2012)
                       | (df_combined_toks.year > 2017)].index)
  df_less_all['toks_pdf2txt'] = df_less_all.toks_pdf2txt.apply(
      lambda x: [] if not len(x) else x)
  df_less_all['toks_metada'] = df_less_all.toks_metada.apply(
      lambda x: [] if not len(x) else x)
  df_less_all['concat_toks'] = df_less_all.apply(
      lambda row: row.toks_metada + row.toks_pdf2txt, axis=1)

  dict_restricted = Dictionary.load(os.path.join(
      DATA_DIR, 'corpora', 'dictionary.less.all'))
  # Create a bow tagging for each publication:
  df_less_all['bow'] = df_less_all['concat_toks'].apply(
      dict_restricted.doc2bow)

  # Generate a corpus based on the tokens, which we will be using later
  corpus = df_less_all.bow.tolist()

  # The function
  coherence_scores = find_topics(
      num_topics, corpus, texts=df_less_all.concat_toks.tolist(),
      dictionary=dict_restricted, store_dir=store_dir)
  print(coherence_scores)
  write_scores(coherence_scores, store_dir)
