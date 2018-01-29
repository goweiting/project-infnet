# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import os, csv, re
from itertools import product
from scrapy.exceptions import DropItem


class PersonPipeline(object):
    """
    For `explorerSpider`: Save all the individual's information from the school of informatics
    """

    def open_spider(self, spider):
        self.file1 = open(os.getcwd() + '/data/peopleOfInformatics.csv', 'w')
        self.file2 = open(os.getcwd() + '/data/personPubPageURL.txt', 'w')

        self.writer1 = csv.writer(self.file1, delimiter=',',
                                  quotechar='"', quoting=csv.QUOTE_ALL)

    def close_spider(self, spider):
        self.file1.close()
        self.file2.close()

    def process_item(self, item, spider):
        """
        Write the items to csv files:
        1) For the collection of personal URL (for publicationSpider);
        2) Database for individuals in School
        """
        row = [item['id'], item['last_name'], item['first_name'],
               item['personal_url'],
               item['organisation']['position'], item['organisation']['parent'], item['organisation']['institute']
               ]
        self.writer1.writerow(row)

        pers_url = item['personal_url']
        self.file2.write("{}/publications.html\n".format(str(pers_url).rsplit('.', 1)[0]))


class AliasDuplicatePipeline(object):
    """
    Remove duplicated alias for an individual;
    Pipeline for the `Alias` items
    """

    def __init__(self):
        self.index = {}

    def open_spider(self, spider):
        self.file1 = open(os.getcwd() + '/data/peopleOfInformatics_ALIAS.csv', 'w')
        self.writer1 = csv.writer(self.file1, delimiter=",",
                                  quotechar='"', quoting=csv.QUOTE_ALL)

    def close_spider(self, spider):
        for id, aliases in self.index.items():
            row = []
            row.append(str(id))
            _aliases = "|".join(list(aliases))
            row.append(_aliases)
            self.writer1.writerow(row)

        self.file1.close()

    def process_item(self, item, spider):
        if item.name == "alias":
            if item['id'] in self.index.keys():
                self.index[item['id']].add(item['alias'])
            else:
                self.index[item['id']] = set()
                self.index[item['id']].add(item['alias'])

        return item


class PubNamePipeline(object):
    def __init__(self):
        self.index = []

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        pass

    def process_item(self, item, spider):
        if item.name == "authorname":
            pub_id = item['pub_id']
            _names = []
            # add something we have not seen to the index
            if pub_id not in self.index:
                for name in item['names']:
                    _names.extend(re.findall(r'(\w+, [\w+. ]*)', name))
                self.index.append(pub_id)
            else:
                # we dont have to process the same document again!
                DropItem()

            item['names'] = '|'.join(_names)
        return item


class PubPagePipeline(object):
    """
    remove duplication of publications that have already been seen:
    """

    def __init__(self):
        self.publications = {}

    def open_spider(self, spider):
        self.file1 = open(os.getcwd() + '/data/publications.csv', 'w')
        self.writer1 = csv.DictWriter(self.file1,
                                      ['pub_id', 'date', 'year', 'title', 'authors', 'pub_url',
                                       'doi_url', 'pdf_url', 'abstract', 'publications'])
        self.file2 = open(os.getcwd() + '/data/edges_allowDUP.csv', 'w')
        self.writer2 = csv.writer(self.file2, delimiter=",",
                                  quotechar='"', quoting=csv.QUOTE_ALL)

    def close_spider(self, spider):
        self.writer1.writeheader()

        for id, item in self.publications.items():
            self.writer1.writerow(item)
        self.file1.close()
        self.file2.close()

    def process_item(self, item, spider):
        if item.name == 'publication':
            # set the id for the publication:
            id = item['pub_id']

            if id not in self.publications.keys():
                # add new publication to the dictionary
                self.publications[id] = item

                # add the publication into the collaboration network:
                authors = item['authors']  # a list of authors
                authors = authors.split("|")
                edges = []
                for pair in product(authors, authors):
                    if pair[0] != pair[1] and (pair[1], pair[0]) not in edges:
                        # add to the list of edges where two unique individuals exists
                        # Here, we do not check if the edge pair has existed
                        edges.append(pair)

                for i in edges:
                    self.writer2.writerow([i[0], i[1]])
            else:
                DropItem()

        return item


class AliasSearcherPipeline(object):
    """
    Whats left from here are unique publication's detail:
    those that are scrapped from the individual's page and those from the publication page itself.
    so we map the alias with the similar publication details extracted
    """
    def __init__(self):
        # name - alias pair
        self.index = {}
        self.ALIASseenpub_id = []
        self.PUBseenpub_id = []

    def open_spider(self, spider):
        self.file1 = open(os.getcwd() + '/data/pubpage_aliases.csv', 'w')
        self.writer1 = csv.writer(self.file1, delimiter=',',
                                  quotechar='"', quoting=csv.QUOTE_ALL)

    def close_spider(self, spider):
        for i, v in self.index.items():
            self.writer1.writerow([i, v['full_name'], v['alias']])
        self.file1.close()

    def process_item(self, item, spider):
        if item.name in ['publication', 'authorname']:
            pub_id = item['pub_id']
            if pub_id not in self.index.keys():
                self.index[pub_id] = {}
        else:
            DropItem()

        if item.name == 'publication':
            # Given a publication id that is not seen, we extract the names
            if item['pub_id'] not in self.PUBseenpub_id:
                self.index[item['pub_id']]['full_name'] = item['authors']
                self.PUBseenpub_id.append(item['pub_id'])

        elif item.name == 'authorname':
            if item['pub_id'] not in self.ALIASseenpub_id:
                self.index[item['pub_id']]['alias'] = item['names']
                self.ALIASseenpub_id.append(item['pub_id'])

        return item
