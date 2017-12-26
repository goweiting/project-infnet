# Informatics Collaboration Network & Topic Network
Focusing on the School of Informatics, University of Edinburgh, a collaboration network was created using informtion from the University's collection of research publications [Edinburgh Research Explorer](http://www.research.ed.ac.uk/portal/en/organisations/school-of-informatics(d9a3581f-93a4-4d74-bf29-14c86a1da9f4).html "School of Informatics - Edinburgh Research Explorer"). More details in [infnet-scrapper](./infnet-scrapper/notebooks).

Using the publications (abstracts) scrapped from the research explorer, [topic modells](./topicModel/notebooks) were inferred, and a topic network[1] was generated. A collaboration network was also [created, visualised and analysed](./infnet-analysis/notebooks).

---

## Setting up
The project is still in development. To use the datasets and run the notebooks on your system, follow the following instruction:

1. The project is developed in python2.7. Using anaconda to setup the virtual environment will be the easiest. You can get a copy of miniconda by issuing the following command:
```bash
$ curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh # For MacOSX
$ curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh # For linux/ubuntu
$ bash Miniconda3-latest-MacOSX-x86_64.sh # Install miniconda onto your system
$ echo "export PATH=\""\$PATH":$HOME/miniconda3/bin\"" >> ~/.benv
$ source ~/.benv

# Create conda environment (name infnet) for project:
$ conda create -n infnet python=2.7
$ source activate infnet
```

Now, we have to install required python packages. This list is **updated** as the project progresses:

0. In general, for viewing the jupyter notebooks and for plotting the figures:
```
$ conda install jupyter
$ conda install matplotlib seaborn
```

1. For [infnet-scrapper](./infnet-scrapper):
```
$ conda install scrapy # for scrapping the research explorer
$ conda install pandas # for pre-processing of the scrapped data
```

2. For [infnet-analysis](./infnet-analysis):
```
$ conda install networkx matplotlib numpy
$ pip install powerlaw # for estimating the distribution of degrees in network
$ pip install python-louvain # community detection package
```

3. For [topic modelling](./topicModel):
Before we input the publications into LDA, there is a need to apply ***standard NLP techniques pre-processing*** to create the Bag-of-words (BOW) representation of the text data: `tokenize -> stopping -> stemming`

To that end, the following packages are required:
```
$ pip install pystemmer
```

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
---

## Todo:
1. CS Ranking - create similar collaboration network using data from DBLP; comparing collaboration networks, and if possible, the topic network too

## Credits
[1]: D. M. Blei and J. D. Lafferty, “A correlated topic model of Science,” The Annals of Applied Statistics, vol. 1, no. 1, pp. 17–35, Jun. 2007.



