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

passes = 10  # total number of times the corpus is seen
iterations = 1000  # how many times each document is seen
chunksize = 2000  # how many documents each mini-batch
update_every = 0  # Batch learning
eval_every = None
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
  DATA_DIR = '../../../data/data_dblp/'
  # Parameters FOR LDA:
  # num_topics = 35 ### SEE THE R-notebook
  num_topics = [20,30,40,50,60,70,80,90,100,120,140]
    
  # Import the dataset:
  dblp_toks = pd.read_pickle(os.path.join(DATA_DIR,'toks','toks.dblp.pkl'))
  dict_dblp = Dictionary.load(os.path.join(DATA_DIR, 'corpora', 'dictionary.dblp.1997-2017'))
  
  # convert dblp_toks to BOW:
  dblp_toks['bow'] = dblp_toks['toks'].apply(dict_dblp.doc2bow)  
  corpus_dblp = dblp_toks.bow.tolist()
  
  coherence_scores = find_topics(num_topics, corpus_dblp, texts=dblp_toks.toks.tolist(), dictionary=dict_dblp)
  pkl.dump(coherence_scores, open('./coherencescores.pkl','wb'))
