# data (was infnet-scrapper)
contains two components:

1. [bin](./bin): scripts and binaries for:
    1. Data mining using [scrapy](https://github.com/scrapy/scrapy).
    2. Converting PDFs downloaded to text files using [pdfminer.six](https://github.com/pdfminer/pdfminer.six)
    3. and more scripts!

2. [Notebook](./notebooks) For:
    1. Preprocessing the scrapped data for used - this mainly concern cleaning up the different aliases that a specific individual might have across different publications.
    2. Notebooks for each different dataset is isolated for ease of reference.
3. Dataset: We are interested in the collaboration network that we can derieve from [Edinburgh Research Explorer](http://www.research.ed.ac.uk/portal/). In specific, we are interested in the reserach outputs (publications) from the School of Informatics.
    1. [data:schoolofinf](./data_schoolofinf) Data scrapped from School of Informatic's EDinburgh Research Explorer.
    2. DBLP and arXiv dataset (not available on Github)

4. In each of the dataset, original scrapped documents are available, as well as other metadata. Also included are the processsed `toks` for topic modelling.


### Versions of data scrapped:
- 15 Nov 2017
  - First upload of data scrapped from School of Informatics
- 17 Jan 2018
  - Second upload of data scrapped from School of Informatics
  - Move first dataset to [`data_old`](data_old)
  - Also scrapped School of Mathematics. See [`data_schoolofmathematics`](data_schoolofmathematics)
- 28 Jan 2018
  - modified `infnet-scrapper` to `data`; Reorganisation for clarity
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

### Output from infnet-scrapper
The output from this module are the following pandas dataframe that is pickled (also in csv) for usage by future modules:

1. lookup_poinf.pkl(.csv)     :: information for `poinf`
2. lookup_pub.pkl(.csv)       :: information for `publication`s scrapped
3. institutes.pkl             :: the different institutes saw in `lookup_poinf`

<!-- To prevent overwriting of these files, modules that uses these data such be added to respective `data` folder using symbolic link:

```bash
# To link lookup_pub.pkl from infnet-scrapper/data to infnet-analysis/data:
$ pwd
~/project/infnet-analysis/data
$ ln -s ~/project/infnet-scrapper/data/lookup_pub.pkl . # creates a symbolic of lookup_pub.pkl here:
$ l | grep lookup_pub.pkl # to confirm that the symbolic link is created correctly:
....lookup_pub.pkl -> ~/project/infnet-scrapper/data/lookup_pub.pkl
``` -->


## Todo:
<s>Check out CSRanking for comparisons</s>
[ ] Try python package for [nameparser](https://pypi.python.org/pypi/nameparser/0.5.4) so that more accurate alias-ing can be gathered.
