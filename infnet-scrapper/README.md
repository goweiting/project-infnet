# infnet-scrapper
contains two components:

1. Scrapping data using [scrapy](https://github.com/scrapy/scrapy).
2. Preprocessing the scrapped data for used - this mainly concern cleaning up the different aliases that a specific individual might have across different publications.


---

## To run `scrapy`

0. Activate the `infnet` environment that was setup [here](../README.md)
1. From the command-line:
```
$ pwd
~/project/infnet-scrapper
$ scrapy list # preview all the spiders created:
peopleSpider
publicationSpider
```
2. To find all the individuals in the school - `people-of-informatics` (`poinf`), we call **peopleSpider** to crawl:
```
$ scrapy crawl peopleSpider
```
This generates all the individuals' publication page in [personPubPageURL.txt](data/personPubPageURL.txt), as well as the details of each individual in [peopleOfInformatics.csv](data/peopleOfInformatics.csv).

3. Now, we call **publicationSpider**, to visit all the individual's publication page, mining all their publications:
```
$ scrapy crawl publicationSpider
```
This futher generates:

1. [publications.csv](data/publications.csv) for all the publications visited; duplicated publications are removed. Each publication have an unique id as in the url

2. [pubpage_aliases.csv](data/pubpage_aliases.csv) where the aliases are observed in the each of the publication page

3. [peopleOfInformatics_ALIAS.csv](data/peopleOfInformatics_ALIAS.csv) where the aliases of each paper, with respect to the list of publication from an individuals' list of publication is seen

## Preprocessing **raw** data
Although scrapy had preprocessed some of the raw data it scrapped - for instance publications that have been seen are removed - this is insufficient. The notebook, [preprocess_poinf.ipynb](notebooks/preprocess_poinf.ipynb), hence saw the exploration and preprocessing of the raw data for future usage.

Due to the problem that different alias was used for different publications. In addition, ground-truth institute labels for each individual for comparison is required for network analysis, which was fuzzy as some of the data points were missing/misallocated.

Preview the notebook [here](notebooks/preprocess_poinf.ipynb)

