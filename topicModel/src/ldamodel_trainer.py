"""
individual files call ldamodel_trainer to generate each model;
functions defined in this file is generic
"""

import csv
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')
from numpy.random import RandomState
from gensim import models
# Use coherence model to measure the LDA models generated
from gensim.models.coherencemodel import CoherenceModel

# passes = 100  # total number of times the corpus is seen
# iterations = 1000  # how many times each document is seen
passes = 100  # total number of times the corpus is seen
iterations = 1000  # how many times each document is seen
chunksize = 2000  # how many documents each mini-batch
update_every = 2000  # Batch learning
eval_every = 2000
rng = RandomState(93748573)


def find_topics(num_topics, corpus, texts, dictionary, store_dir):
  """
  Given a list topics, compute the coherence score for each and output the best LDA model.
  """
  coherence_scores = []
  __ = dictionary[0] # temporary load the dictionary into memory so that id2token is available
  id2word = dictionary.id2token
  for num_topic in num_topics:
    print(('TRAINING LDA with {} topics'.format(num_topic)))
    # This is the fullpub LDA model.
    ldamodel = models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, eta='auto', alpha='auto', num_topics=num_topic,
                                        iterations=iterations, passes=passes, update_every=update_every, eval_every=eval_every, random_state=rng)

    # Compute the C_V coherence score
    cm = CoherenceModel(ldamodel, texts=texts,
                        dictionary=dictionary, coherence='c_v')
    score_cv = cm.get_coherence()
    cm = CoherenceModel(ldamodel, corpus=corpus,
                        dictionary=dictionary, coherence='u_mass')
    score_umass = cm.get_coherence()
    print(('c_v: {:4f}\tu_mass{:4f}'.format(score_cv, score_umass)))

    coherence_scores.append(
        {'num_topic': num_topic, 'c_v': score_cv, 'u_mass': score_umass})
    ldamodel.save('{}/ldamodel_nb_topics_{}'.format(store_dir, num_topic))
  return(coherence_scores)


def write_scores(coherence_scores, store_dir):
  with open(store_dir + '/scores.csv', 'w') as f:
    fieldnames = ['num_topic', 'c_v', 'u_mass']
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    writer.writerows(coherence_scores)
