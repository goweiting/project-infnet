"""
Installation for NLTK:
in a new terminal:
$ source activate infnet3
$ python
>> import nltk
>> nltk.download('stopwords')
>> nltk.download('wordnet')
"""

# Packages for standard Pre-processing:
import string
from nltk.corpus import stopwords # Common stopwords 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

# # For generation of Bi-grams and tri-grams
# from gensim.models.phrases import Phraser, Phrases

### Functions for simple preprocessing of summary:
sw = stopwords.words('english')
sw.extend(list(string.punctuation))
stop = set(sw)

# tokenizing the sentences; this uses punkt tokenizer
tokenizer = RegexpTokenizer(r'\w+')
tokenize = lambda x : tokenizer.tokenize(x)

# apply stopping, and remove tokens that have length of 1
removeSW = lambda x : list([t.lower() for t in x if t not in stop and len(t) > 1 and t.isalpha()])

# Lemmatizing
lemmatizer = WordNetLemmatizer()
lemmify = lambda x : [lemmatizer.lemmatize(t) for t in x]

preprocess = lambda x: lemmify(removeSW(tokenize(x)))
