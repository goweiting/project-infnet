import scrapy
import re, os
from scrapy.crawler import CrawlerProcess
from scrapy.loader import ItemLoader
from scrapy.loader.processors import TakeFirst, MapCompose, Join, Compose

# Global function to remove items except for alphanumeric
rmUnwantedChars = lambda x: (re.sub(r'[^ a-zA-Z]', '', x)).strip().lower()

with open(os.getcwd() + '/data/personPubPageURL.txt') as f:
    pubURL = f.readlines()
    pubURL = map(lambda x: x.strip(), pubURL)


class Alias(scrapy.Item):
    name = "alias"
    id = scrapy.Field(
        input_processor=MapCompose(lambda x: x.encode('utf-8')),
        output_processor=TakeFirst()
    )
    alias = scrapy.Field(
        input_processor=MapCompose(lambda x: x.encode('utf-8').lower()),
        output_processor=TakeFirst()
    )


class AliasLoader(ItemLoader):
    default_item_class = Alias


class AuthorName(scrapy.Item):
    name = "authorname"
    pub_id = scrapy.Field(
        output_processor=TakeFirst()
    )
    names = scrapy.Field(
        input_processor=MapCompose(lambda x: x.encode('utf-8').lower().strip()),
    )


class AuthorNameLoader(ItemLoader):
    default_item_class = AuthorName


class Publication(scrapy.Item):
    name = "publication"
    title = scrapy.Field(
        input_processor=MapCompose(rmUnwantedChars),
        output_processor=TakeFirst()
    )
    authors = scrapy.Field(
        input_processor=MapCompose(lambda x: x.lower()),
        output_processor=Join(separator='|')
    )
    date = scrapy.Field(
        input_processor=MapCompose(lambda x: x.encode('utf-8')),
        output_processor=TakeFirst()
    )
    pub_url = scrapy.Field(
        output_processor=TakeFirst()
    )
    doi_url = scrapy.Field(
        output_processor=TakeFirst()
    )
    abstract = scrapy.Field(
        input_processor=MapCompose(str, lambda x: x.strip().lower()),
        output_processor=TakeFirst()
    )
    pdf_url = scrapy.Field(
        output_processor=TakeFirst()
    )
    pub_id = scrapy.Field(
        output_processor=TakeFirst()
    )
    publications = scrapy.Field(
        output_processor=Join(separator=',')
    )
    year = scrapy.Field(
        output_processor=TakeFirst()
    )


class PublicationLoader(ItemLoader):
    default_item_class = Publication


class CollaborationNetworkSpider(scrapy.Spider):
    custom_settings = {
        'ITEM_PIPELINES': {
            'scrapper.pipelines.AliasDuplicatePipeline': 300,
            'scrapper.pipelines.PubPagePipeline': 400,
            'scrapper.pipelines.PubNamePipeline': 700,
            'scrapper.pipelines.AliasSearcherPipeline': 800
        }
    }
    name = "publicationSpider"
    urls = pubURL

    # urls = [
    #     """http://www.research.ed.ac.uk/portal/en/persons/rik-sarkar(4d5e5fe2-2528-4d9b-b959-040eda566c1b)/publications.html"""]

    def start_requests(self):
        try:
            for url in self.urls:
                yield scrapy.Request(url=url, callback=self.parse)
        except:
            pass

    def parse(self, response):
        # go through each sub item listed on the publication page:
        # First, retrieve the individual's id  from the url of the individual's page:
        url = response.url
        person_id = re.findall(r'\((.*)\)', url)[0]

        # Now, go through each of the individual's listed item in the portal:
        for sel in response.xpath('//li[@class="portal_list_item"]'):
            # Find the publication url and scrap from the url instead:
            pub_link = sel.xpath('.//h2/a[@class="link"]/@href').extract_first()
            yield scrapy.Request(url=pub_link, callback=self.parsePubPage)

            # relate the authors's short name with a particular project (based on the project id)
            # this allows us to link up the author's full name in the final page:
            texts = sel.xpath('.//text()').extract()
            # the names lies between title of the publication and the date:
            date = sel.xpath('.//span[@class="date"]/text()').extract_first().encode('utf-8')
            # find the indices for these elements
            _date_idx = texts.index(date)
            # whatever between must be the names
            names = texts[1: _date_idx]
            pub_id = re.findall(r'\((.*)\)', pub_link)[0].encode('utf-8')
            pnl = AuthorNameLoader()
            pnl.add_value('pub_id', pub_id)
            pnl.add_value('names', names)
            yield pnl.load_item()

            # Individuals in school of informatics will be hyperlinked if they published together
            # Here, we can use this information to find the shortname of the individual by finding if
            # the user_id of the link is the same as the person_id for this page:
            al = AliasLoader()
            al.add_value('id', person_id)
            for i_link in sel.xpath('.//a[@class="link person"]'):
                _link = i_link.xpath('@href').extract_first()
                i_id = re.findall(r'\((.*)\)', _link)[0]
                if i_id == person_id:
                    alias = i_link.xpath('.//span/text()').extract_first()
                    al.add_value('alias', alias)
                    yield al.load_item()

        # Basically clicks the `next` button
        for sel in response.xpath('//a[@class="portal_navigator_next common_link"]'):
            next = sel.xpath('@href').extract_first()  # extract the link to the next page
            yield scrapy.Request(url=next, callback=self.parse)

    def parsePubPage(self, response):

        pl = PublicationLoader()
        pub_url = response.url
        pl.add_value('pub_url', pub_url)
        pub_id = re.findall(r'\((.*)\)', pub_url)[0]
        pl.add_value('pub_id', pub_id)

        # Authors:
        authors = []
        for sel in response.xpath('//ul[@class="relations persons"]/li'):
            a = sel.xpath('.//text()').extract_first().encode('utf-8')
            authors.append(a)
        pl.add_value('authors', authors)

        # Date and year
        date = response.xpath('//span[@class="date"]/text()').extract_first(default='UNKNOWN')
        pl.add_value('date', date)
        pl.add_value('year', date.split(' ')[-1].encode('utf-8'))

        # Title
        title = response.xpath('//h2[@class="title"]/span/text()').extract_first(default="UNKNOWN").encode('utf-8')
        pl.add_value('title', title)

        # doi_URL
        doi_url = response.xpath(
            '//ul[@class="relations digital_object_identifiers"]//a[@class="link"]/@href').extract_first(
            default='UNKNOWN').encode('utf-8')
        pl.add_value('doi_url', doi_url)

        # Abstract (if present):
        abstract = response.xpath('//div[@class="textblock"]/text()').extract_first(default="UNKNOWN").encode('utf-8')
        pl.add_value('abstract', abstract)

        # pdf link:
        pdf = 'UNKNOWN'
        for link in response.xpath('//a[@class="link"]/@href').extract():
            if link[-4:] == ".pdf":
                pdf = link
                break
        pl.add_value('pdf_url', pdf.encode('utf-8'))

        # publisher/journal information:
        details = []
        for tr_sel in response.xpath('//tr'):
            th = tr_sel.xpath('.//th').extract_first().lower()
            # check if the word journal or publication pr publisher exists in it:
            try:
                if re.findall(r'(journal|publi)', th)[0]:
                    dets = tr_sel.xpath('.//td//text()').extract_first().encode('utf-8').lower()
                    details.append(dets)
            except IndexError:
                continue
        if len(details):
            pl.add_value('publications', details)
        else:
            pl.add_value('publications', 'UNKNOWN')

        yield pl.load_item()


# =====================================================================================
def main():
    depth = 100
    delay = 1

    process = CrawlerProcess({
        #  'USER_AGENT': 'Chrome/58.0',
        'DEPTH_LIMIT': depth,
        'DOWNLOAD_DELAY': delay,
        'CONCURRENT_REQUESTS': 1000
    })

    spider = CollaborationNetworkSpider()
    process.crawl(spider)
    process.start()  # the script will block here until the crawling is finished
    # spider.finalize()


if __name__ == "__main__":
    main()
