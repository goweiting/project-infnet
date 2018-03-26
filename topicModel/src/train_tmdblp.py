#!/usr/bin/env python
import logging
import os
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')
import pickle as pkl
import pandas as pd
from numpy.random import RandomState
rng = RandomState(93748573)
from gensim import models
from gensim.corpora import Dictionary
# Use coherence model to measure the LDA models generated
from gensim.models.coherencemodel import CoherenceModel

# FOR LDA PARAMETERS
passes = 1  # total number of times the corpus is seen BECAUSE it will fault repeatedly! once should be good enough?
iterations = 1000  # how many times each document is seen
chunksize = 2000  # how many documents each mini-batch
update_every = 0  # Batch learning
eval_every = None

DATA_DIR = '../../data/data_dblp'
dblp_toks = pd.read_pickle(os.path.join(DATA_DIR, 'toks', 'toks.dblp.pkl'))
dict_dblp = Dictionary.load(os.path.join(
    DATA_DIR, 'corpora', 'dictionary.dblp.1997-2017'))
# convert dblp_toks to BOW:
dblp_toks['bow'] = dblp_toks['toks'].apply(dict_dblp.doc2bow)
corpus = dblp_toks.bow.tolist()
del(dblp_toks)  # free space by removing the dataframe

# Parameters FOR LDA:
num_topics = 100  # RANDOMLY 100 topics
tmp = dict_dblp[0]
id2word = dict_dblp.id2token
# This is the fullpub LDA model.
lda_dblp = models.ldamulticore.LdaMulticore(
    minimum_probability=.01,
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    eta='auto',
    alpha='symmetric',
    num_topics=num_topics,
    iterations=iterations,
    passes=passes,
    eval_every=eval_every,
    workers=10)  # 10 threads

lda_dblp.save(os.path.join(DATA_DIR,'models','tm','tm_dblp'))
