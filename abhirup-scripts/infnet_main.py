import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
import ast, pickle
import networkx as nx
import os

pub_token_map_filename="data/pub_token_map.csv"
authors_filename="data/authors.csv"
authors_alias_filename="data/authors_alias.csv"
publications_filename="data/publications.csv"

def save_pickle(obj, name):
    with open('data/tmp/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(name):
    with open('data/tmp/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_files():
    pub_token_map = pd.read_csv(pub_token_map_filename)
    pub_token_map.rename(columns={'toks_metada': 'tokens'}, inplace=True)
    authors = pd.read_csv(authors_filename)
    authors_alias = pd.read_csv(authors_alias_filename)
    publications = pd.read_csv(publications_filename)

    return pub_token_map, authors, authors_alias, publications

def get_word_frequencies(pub_token_map):
    dict_1word = {}
    for index, row in pub_token_map.iterrows():
        tokens = ast.literal_eval(row['tokens'])
        for word in tokens:
            if isinstance(word, str):
                unicode_word = unicode(word, 'utf-8')
            else:
                unicode_word = word
            s = dict_1word.get(unicode_word, np.array([]))
            dict_1word[unicode_word] = np.append(s, row['pub_id'])
        if (index % 100==0):
            print index
    save_pickle(dict_1word, 'dict_1word')

# Not sure if this is right. As high frequency words are potential query words too.
def filter_high_freq_dict_1words():
    dict_1word = load_pickle("dict_1word")
    sorted_1word_freq = load_pickle("sorted_1word_freq")
    arr = np.array([float(t[1]) for t in sorted_1word_freq])
    high_freq_threshold = np.percentile(arr, 95)
    print  high_freq_threshold, [sorted_1word_freq[i] for i in np.where(arr>high_freq_threshold)[0] ]
    #for key in sorted_1word_freq

def analyze_word_frequencies():
    dict_1word = load_pickle('dict_1word')
    word_freq=[]
    for word in dict_1word.keys():
        word_freq.append( (word, len(np.unique(dict_1word[word]))) )
    sorted_word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    save_pickle(sorted_word_freq, 'sorted_1word_freq')
    sorted_word_freq = load_pickle('sorted_1word_freq')
    print sorted_word_freq[:50]

'''
def get_all_pair_frequencies():
    dict_2word = {}
    dict_1word = load_pickle('dict_1word')
    i=0
    for word1 in dict_1word.keys():
        pubs1 = dict_1word[word1]
        if (len(pubs1) <= 1):
            continue
        for word2 in dict_1word.keys():
            pubs2 = dict_1word[word2]
            if (len(pubs2) <= 1):
                continue
            if (word1 == word2):
                continue
            key = word1+'.'+word2
            s = dict_2word.get(key, [])
            if (len(s) == 0):
                key = word2+'.'+word1
                s = dict_2word.get(key, [])
            if (len(s) != 0):
                continue
            xs = set(pubs1).intersection(set(pubs2))
            #if (len(xs) > 0 ):
            #    print word1, "pubs1:", len(pubs1),
            #    print word2, "pubs2:", len(pubs2),
            #    print "intersection:", len(xs)
            if (len(xs) > 0):
                dict_2word[key] = xs
        if (i%10 == 0):
            print i
            save_pickle(dict_2word, str(i)+'_dict_2word')
            dict_2word = {}
        i=i+1
    save_pickle(dict_2word, 'dict_2word')
'''

def get_pub_token_map_dict(pub_token_map):
    d = {}
    for index, row in pub_token_map.iterrows():
        tokens = ast.literal_eval(row['tokens'])
        for word in tokens:
            if isinstance(word, str):
                unicode_word = unicode(word, 'utf-8')
            else:
                unicode_word = word
            s = d.get(row['pub_id'], [])
            s.append(unicode_word)
            d[row['pub_id']] = s
    save_pickle(d, "pub_token_map_dict")

def get_freq_sorted_authors(output_authors):
    uniq_authors = np.array([])
    freq_count = np.array([])
    for x in output_authors:
        if (len(uniq_authors) > 0 and x in uniq_authors):
            index = np.where(uniq_authors == x)[0][0]
            freq_count[index] = freq_count[index] + 1
        else:
            uniq_authors = np.append(uniq_authors, x)
            freq_count = np.append(freq_count, 1)
    additional_list = []
    for i in range(len(uniq_authors)):
        additional_list.append( (uniq_authors[i], freq_count[i] ) )
    return sorted(additional_list, reverse=True, key=lambda t: t[1])

def get_sorted_authors(output_authors_scores):
    authors_aggregate_scores = []
    for key in output_authors_scores.keys():
        total_score = sum(output_authors_scores[key])
        authors_aggregate_scores.append( (key, total_score) )
    authors_aggregate_scores.sort(key=lambda t: t[1], reverse=True)
    return authors_aggregate_scores

def plot_collab_graph(collab_graph):
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        positions = graphviz_layout(collab_graph)
        const = 40.
        xs, ys = [], []
        for key in positions.keys():
            #positions[key] = ( np.random.uniform(positions[key][0]-const, positions[key][0]+const),
            #                   np.random.uniform(positions[key][1]-const, positions[key][1]+const) )
            positions[key] = ( np.random.uniform(positions[key][0]-const, positions[key][0]+const),
                               positions[key][1])
            xs.append(positions[key][0]); ys.append(positions[key][1])
        plt.figure(3,figsize=(15,15))
        nx.draw(collab_graph, pos=positions, with_labels=True, node_color='w', font_size=10, node_size=0)
        #plt.show()
        plt.savefig('static/images/collab_graph.png')
    except ImportError, e:
        try:
            print "Install graphviz_layout to view collaboration graph."
            os.remove('static/images/collab_graph.png')
        except OSError:
            pass

def get_authors_for_query(_pub_token_map, _authors, _authors_alias, _publications, query_str):
    global pub_token_map, authors, authors_alias, publications
    pub_token_map, authors, authors_alias, publications = _pub_token_map, _authors, _authors_alias, _publications
    # Given this query string we want to find out the related tokens.
    # 1. Find the publications that have this token.
    # 2. Find the other tokens that appear in those publications.
    # 3. Score each co-occurring token whose jaccard similarity and confidence both are good.
    # 4. Take tokens with score more than 95 percentile of the scores.
    # 5. Find the set of publications they occur in

    # 6. score of a publication = sum of the scores of the tokens in that publication (tokens which do not relate to query have score 0)
    #       - Another way to identify the important publications is the union of the k-nn of the publications where the query appeared. LSH might be a good option. But LSH on Jaccard similarity is for detecting exactly duplicate documents. But, here the documents are similar not duplicates. So, standard way for LSH does not apply.
    #       - Another way is to find k-nn in the citation graph of the publications where the query appeared.
    # 7. Take publications which score higher than 95 percentile of all the related publications

    dict_1word = load_pickle("dict_1word")
    pubs_query = dict_1word.get(query_str, [])
    if (len(pubs_query) == 0):
        print "This term is unknown to us."
        return [], []
    pub_token_map_dict = load_pickle('pub_token_map_dict')
    tokens = []
    #print "It occured in #of pubs:", len(pubs_query)
    for pub in pubs_query:
        ts = pub_token_map_dict[pub]
        #print "publication:", pub, "tokens:", ts
        [tokens.append(t) for t in ts]
    tokens_arr = np.array(tokens)
    #print "Number of unique tokens:", len(np.unique(tokens_arr)), len(tokens)
    data = []
    for t in np.unique(tokens_arr):
        pubs_t = dict_1word.get(t, [])
        xs = set(pubs_query).intersection(set(pubs_t))
        us = set(pubs_query).union(set(pubs_t))
        jaccard_sim = len(xs) / float(len(us))
        confidence = len(xs) / float(len(pubs_query))
        data.append( [len(xs), jaccard_sim*confidence, t] )
    sorted_data = np.array(sorted(data, key=lambda x: x[1], reverse=True))
    threshold = np.percentile(sorted_data[:,1].astype('float'), 97)
    #print "90 percentile:", threshold
    chosen_tokens = np.array(sorted_data[np.where(sorted_data[:,1].astype('float')>=threshold)[0]])[:,[1, 2]]
    pubset=set()
    for _, tok in chosen_tokens:
        #print "dict_1word[",tok,"]:", len(dict_1word[tok])
        pubset = pubset.union(set(dict_1word[tok]))
        #print pubset
    #print len(pubset)
    #print "chosen_tokens:", chosen_tokens

    related_pubs = []
    for pub in pubset:
        tokens = pub_token_map_dict[pub]
        related_tokens = set(tokens).intersection(set(chosen_tokens[:,1]))
        #print "related_tokens:", related_tokens
        pub_score=0
        for rel_tok in related_tokens:
            #print "np.where(chosen_tokens[:,1] == rel_tok):", chosen_tokens[np.where(chosen_tokens[:,1] == rel_tok)[0]][0][0]
            pub_score += float(chosen_tokens[np.where(chosen_tokens[:,1] == rel_tok)[0]][0][0])
            #print "related_token:", rel_tok, "score:", score
        #print pub, "related_tokens:", related_tokens, "pub_score:", pub_score
        related_pubs.append( [pub, pub_score, related_tokens] )

    sorted_pubs = np.array(sorted(related_pubs, key=lambda x: x[1], reverse=True))
    threshold = np.percentile(sorted_pubs[:,1].astype('float'), 95)
    chosen_pubs = np.array(sorted_pubs[np.where(sorted_pubs[:,1].astype('float')>=threshold)[0]])
    #print chosen_pubs

    #print "chosen_pubs:", chosen_pubs
    #chosen_pubs = [pub_id, score, set of chosen tokens]
    #### Now extract the authors from the publication and author list. ####
    output_authors = []
    output_authors_scores = {} # Output authors with pub scores
    collab_graph = nx.Graph()
    for pub in chosen_pubs:
        pub_authors = list(publications[publications['pub_id'].str.match(pub[0])==True]['authors'])[0].split('|')
        pub_score = pub[1]
        full_names = []
        for author in pub_authors:
            guess_author_name = max(author.split(' '), key=len)
            guess_author_ids = list(authors_alias[authors_alias['alias'].str.contains(guess_author_name)==True]['author_id'])
            if (len(guess_author_ids) > 0):
                guess_author_id = guess_author_ids[0]
                name = authors[authors['author_id'].str.contains(guess_author_id)==True][['first_name','last_name']].values[0]
                full_name = name[0]+ " " +name[1]
                output_authors.append(full_name)
                xscores = output_authors_scores.get(full_name, [])
                xscores.append(pub_score)
                output_authors_scores[full_name] = xscores
                full_names.append(full_name)

        if (len(full_names) == 1):
            collab_graph.add_node(full_names[0])
        #iterate over all pairs of authors
        for i in range(len(full_names)):
            for j in range(i+1, len(full_names)):
                a1 = full_names[i]; a2 = full_names[j]
                #print a1, a2
                collab_graph.add_edge(a1, a2)

    plot_collab_graph(collab_graph)

    sorted_authors = get_sorted_authors(output_authors_scores)
    #sorted_uniq_authors = get_freq_sorted_authors(output_authors)
    #print "len(sorted_authors):", len(sorted_authors)
    return chosen_tokens, sorted_authors

### uncomment these to run from the console ###
#pub_token_map, authors, authors_alias, publications = load_files()
#chosen_tokens, sorted_uniq_authors = get_authors_for_query(pub_token_map, authors, authors_alias, publications, "neural_network")
#print sorted_uniq_authors

