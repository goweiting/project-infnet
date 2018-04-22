## Embedding

This module embeds the individuals from informtics collaboration network into topic space. This is achieved by:
1. Create a mapping of publications and individuals ID (id <-> publication)
2. Concatenating tokens used in publication is concatenated (concated_toks)
3. inference of topics distribution is done on the concated_toks
4. The distribution of topics is analogus to a vector space model, where this space is the topic space (much smaller dimension than the length of vocabulary)!


### Directory:
Jupyter notebook are used to create topic-similarity networks
- [notebooks](./notebooks) :: where all the files are as we use notebooks to visualise
  - [embedding_helper.py](./notebooks/embedding_helper.py) :: defines helper function useful for creating topic-similarity networks. These functions are defined in the report
  - [embedding.ipynb](./notebooks/embedding.ipynb) :: where all the topic models are created and visualised
  - two additional notebooks used to explore the communities using the embedding (miscellaneous work):
    - [topicDist_poinf](./notebooks/topicDist_poinf.ipynb)
    - [topicDist_pub](./notebooks/topicDist_pub.ipynb)
  - IMG :: contains all the images generated
  - res :: contains topic-similarity networks represented as adjacency matrix.
