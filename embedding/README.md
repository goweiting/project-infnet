## Embedding

This module embeds the individuals from informtics collaboration network into topic space. This is achieved by:
1. Create a mapping of publications and individuals ID (id <-> publication)
2. Then all the tokens of that publication is concatenated (concated_toks)
3. inference of topics distribution is done on the concated_toks 
4. The distribution of topics is analogus to a vector space model, where this space is the topic space (much samller dimension than the length of vocabulary)!

