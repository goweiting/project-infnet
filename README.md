# Informatics Collaboration Network & Topic Network
Focusing on the School of Informatics, University of Edinburgh, a collaboration network was created using informtion from the University's collection of research publications [Edinburgh Research Explorer](http://www.research.ed.ac.uk/portal/en/organisations/school-of-informatics(d9a3581f-93a4-4d74-bf29-14c86a1da9f4).html "School of Informatics - Edinburgh Research Explorer"). More details in [infnet-scrapper](./infnet-scrapper/notebooks).

Using the publications scrapped from the research explorer, [topic models](./topicModel/notebooks) were inferred, and a topic-similarity networks[1] were generated. A collaboration network was also [created, visualised and analysed](./infnet-analysis/notebooks).

---
## Directory
1. Data
    - bin
        - scrapy : scripts for scraping using scrapy
        - pdfminer: contains binary from [pdfminer.six](https://github.com/pdfminer/pdfminer.six)
        - scripts used to process PDFs using pdfminer
    -  data_dblp : dblp dataset, but metadata of publications are not stored due to the size of the dataset. We only store tokenised pickled files and dictionary in it.
    - data_schoolofinf : Informatics dataset retrieved in Jan 2018
    - notebooks : corresponds to steps taken to process and generate lookup tables for the remaining steps.

2. infnet-analysis
    - notebooks : contain the jupyter notebook used to generate each informatics network.
        - community detection and homophily test is carried out in [analysis.ipynb](infnet-analysis/notebooks/analysis.ipynb)

3. embedding
    - notebooks : creation of topic-similarity networks

4. topicModel
    - notebooks : generate topic models using Gensim's implementation of LDA; also explore the performance of each model
    - src : contain scripts to generate each topic model

## Setting up
The project is still in development. To use the datasets and run the notebooks on your system, follow the following instruction:

1. The project is developed in python3.6. Using anaconda to setup the virtual environment will be the easiest. You can get a copy of miniconda by issuing the following command:
```bash
$ curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh # For MacOSX
$ curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh # For linux/ubuntu
$ bash Miniconda3-latest-MacOSX-x86_64.sh # Install miniconda onto your system
$ echo "export PATH=\""\$PATH":$HOME/miniconda3/bin\"" >> ~/.benv
$ source ~/.benv
```

*** NOTE the use of python3 instead ***
```
# Create conda environment (name infnet3) for project:
# Also install essential packages across all modules:
$ conda create -n infnet3 python=3 pandas matplotlib jupyter ipython ipykernel
$ source activate infnet3 # Activates the environment
(infnet3) $ <--- this shows the successfull acitvation of the environment.
```

Now, we have to install required python packages. This list is **updated** as the project progresses:

1. For [data](./data) pre-processing, additional packages are installed:
```
(infnet3) $ conda install scrapy # for scrapping the research explorer
(infnet3) $ conda install nltk # this is used for creating tokens for topic modelling
```

1a. To configure NLTK, executing the following in a ***new terminal with infnet3 activated*** :
```bash
(infnet3) $ python to launch a python3 shell
> import nltk
> nltk.download('stopwords') # select `yes` when prompt.
> nltk.download('WordNet')
```


2. For [infnet-analysis](./infnet-analysis):
```
(infnet3) $ conda install networkx numpy
(infnet3) $ pip install python-louvain # community detection package
```
<!-- $ pip install powerlaw # for estimating the distribution of degrees in network -->


3. For [topic modelling](./topicModel):
<!-- Before we input the publications into LDA, there is a need to apply ***standard NLP techniques pre-processing*** to create the Bag-of-words (BOW) representation of the text data: `tokenize -> stopping -> stemming`

To that end, the following packages are required:
```
$ pip install pystemmer
```

 -->

For topic modelling using ***latent diriclet allocation***
```
$ conda install gensim # to generate LDA
$ pip install pyldavis # for visualisation of the LDA
```

For data exploration, visualisation of data and clustering:
```
$ conda install scikit-learn # for k-means, manifold, dbscan...
$ conda install -c conda-forge hdbscan
```

<!-- ## Express setup
If you already have conda installed, you can use the `environment.yml` file to setup the infnet experiment:
```bash
$ conda env create -n infnet -f environment.yml
```-->

---
