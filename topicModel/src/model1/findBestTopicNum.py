import logging, os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')
import pickle as pkl
import pandas as pd
import numpy as np
from numpy.random import RandomState
from gensim import models
from gensim.corpora import Dictionary
# Use coherence model to measure the LDA models generated
from gensim.models.coherencemodel import CoherenceModel

passes = 100  # total number of times the corpus is seen
iterations = 1000  # how many times each document is seen
chunksize = 2000  # how many documents each mini-batch
update_every = 2000  # Batch learning
eval_every = 2000
rng = RandomState(93748573)


def find_topics(num_topics, corpus, texts, dictionary):
  """
  Given a list topics, compute the coherence score for each and output the best LDA model.
  """
  model = []
  coherence_scores = []
  tmp = dictionary[0]
  id2word = dictionary.id2token
  for num_topic in num_topics:
    print(('TRAINING LDA with {} topics'.format(num_topic)))
    # This is the fullpub LDA model.
    ldamodel = models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,chunksize=chunksize,eta='auto',alpha='auto',num_topics=num_topic,iterations=iterations,passes=passes,update_every=update_every, eval_every=eval_every, random_state=rng)

    # Compute the C_V coherence score
    cm = CoherenceModel(ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
    score_cv = cm.get_coherence()
    cm = CoherenceModel(ldamodel, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    score_umass = cm.get_coherence()
    print(('c_v: {:4f}\tu_mass{:4f}'.format(score_cv, score_umass)))

    model.append(ldamodel)
    coherence_scores.append({'num_topic':num_topic, 'c_v': score_cv, 'u_mass': score_umass})
    ldamodel.save('./{}_model2_LDAmodel'.format(num_topic))
  return(coherence_scores)

if __name__=='__main__':
  DATA_DIR = '../../../data/data_schoolofinf/'
  # Parameters FOR LDA:
  num_topics = [10,15,20,25,30,35,40,50]
    
  # Import the dataset:
  df_combined_toks = pd.read_pickle(os.path.join(DATA_DIR,'toks', 'toks.combined.pkl'))
  df_combined_toks.head(3)
  df_combined_toks = df_combined_toks.drop(df_combined_toks[(df_combined_toks.year < 1997) | (df_combined_toks.year > 2017)].index)
  df_combined_toks['toks_pdf2txt'] = df_combined_toks.toks_pdf2txt.apply(lambda x: [] if not len(x) else x)
  df_combined_toks['toks_metada'] = df_combined_toks.toks_metada.apply(lambda x: [] if not len(x) else x)


  dict_all = Dictionary.load(os.path.join(DATA_DIR,'corpora','dictionary.all'))
  df_all = df_combined_toks.copy()
  df_all['concat_toks'] = df_combined_toks.apply(lambda row: row.toks_metada + row.toks_pdf2txt, axis=1)
  # Create a bow tagging for each publication:
  df_all['bow'] = df_all['concat_toks'].apply(dict_all.doc2bow)

  # Generate a corpus based on the tokens, which we will be using later
  corpus_all = df_all.bow.tolist()

  # The function
  coherence_scores = find_topics(num_topics, corpus_all, texts=df_all.concat_toks.tolist(), dictionary=dict_all)
  pkl.dump(coherence_scores, open('./coherencescores.pkl','wb'))
