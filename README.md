# Informatics Collaboration Network and Topic Network
Focusing on the School of Informatics, University of Edinburgh, a collaboration network was created using informtion from the University's collection of research publications [Edinburgh Research Explorer](http://www.research.ed.ac.uk/portal/en/organisations/school-of-informatics(d9a3581f-93a4-4d74-bf29-14c86a1da9f4).html "School of Informatics - Edinburgh Research Explorer"). More details in [infnet-scrapper](./infnet-scrapper/notebooks).

Using the publications (abstracts) scrapped from the research explorer, [topic modells](./topicModel/notebooks) were inferred, and a topic network[1] was generated. A collaboration network was also [created, visualised and analysed](./infnet-analysis/notebooks).

## Setting up
The project is still in development. To use the datasets and run the notebooks on your system, follow the following instruction:

1. The project is developed in python2.7. Using anaconda to setup the virtual environment will be the easiest. You can get a copy of miniconda by issuing the following command:
```bash
$ curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh # For MacOSX
$ curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh # For linux/ubuntu
$ bash Miniconda3-latest-MacOSX-x86_64.sh # Install miniconda onto your system
$ echo "export PATH=\""\$PATH":$HOME/miniconda3/bin\"" >> ~/.benv
$ source ~/.benv

# Create conda environment (name py27) for project:
$ conda create-n py27 python=2.7
$ source activate py27
$ conda install networkx gensim jupyter
```

## Todo:
[1] CS Ranking - create similar collaboration network using data from DBLP; comparing collaboration networks, and if possible, the topic network too

## Credits
[1]: D. M. Blei and J. D. Lafferty, “A correlated topic model of Science,” The Annals of Applied Statistics, vol. 1, no. 1, pp. 17–35, Jun. 2007.



