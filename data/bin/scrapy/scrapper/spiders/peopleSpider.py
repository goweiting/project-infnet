import scrapy
import re
from scrapy.crawler import CrawlerProcess
from scrapy.loader import ItemLoader
from scrapy.loader.processors import TakeFirst, MapCompose, Join

# Global function to remove items except for alphanumeric
rmUnwantedChars = lambda x: (re.sub(r'[^ a-zA-Z]', '', x)).strip().lower()


class Person(scrapy.Item):
    id = scrapy.Field()
    personal_url = scrapy.Field()
    organisation = scrapy.Field()
    first_name = scrapy.Field(
        input_processor=MapCompose(lambda x: x.strip().lower()),
        output_processor=TakeFirst()
    )
    last_name = scrapy.Field(
        input_processor=MapCompose(lambda x: x.strip().lower()),
        output_processor=TakeFirst()
    )


class PersonLoader(ItemLoader):
    default_item_class = Person
    default_output_processor = TakeFirst()


class ResearchExplorerSpider(scrapy.Spider):
    name = "peopleSpider"
    custom_settings = {
        'ITEM_PIPELINES': {
            'scrapper.pipelines.PersonPipeline': 300
        }
    }

    urls = [
        #"""http://www.research.ed.ac.uk/portal/en/organisations/school-of-informatics(d9a3581f-93a4-4d74-bf29-14c86a1da9f4)/persons.html"""]
	"""http://www.research.ed.ac.uk/portal/en/organisations/school-of-mathematics(65c8fa0b-21ee-49f2-ae45-1690a020a962)/persons.html"""]

    def start_requests(self):
        try:
            for url in self.urls:
                yield scrapy.Request(url=url, callback=self.parsePerson)
        except:
            pass

    def parsePerson(self, response):
        """ Extract the particulars of individual from School of Infromatics
        Each item in the portal_list_item is a information about the the individual.
        The following information are scrapped:
        [+] the url of the individual
        [+] the unique id that the research explorer uses
        [+] first name, last name
        [+] the organisation that individual belongs to and his/her role in it
        """
        for sel in response.xpath('//li[@class="portal_list_item"]'):
            persLoader = PersonLoader(selector=sel)
            persLoader.add_xpath('personal_url', './/a[@class="link person"]/@href')
            persLoader.add_value('id',
                                 re.findall(r'\((.*)\)', persLoader.get_output_value('personal_url'))[0])
            full_name = sel.xpath('.//a[@class="link person"]/span/text()').extract_first()
            last_name, first_name = full_name.split(',')
            persLoader.add_value('last_name', last_name)
            persLoader.add_value('first_name', first_name)

            # Scrap all the link_organisation that an individual belongs to:
            orgs_info = {'parent': None, 'institute': None, 'position': None}
            orgs_info['position'] = \
                rmUnwantedChars(sel.xpath('.//span[@class="minor dimmed"]/text()').extract_first(default='UNKNOWN'))

            org = sel.xpath('.//a[@class="link organisation"]')
            parent_org = org[0]

            orgs_info['parent'] = \
                rmUnwantedChars(parent_org.xpath('.//span/text()').extract_first(default='UNKNOWN'))

            try:
                institute = org[1]
                orgs_info['institute'] = \
                    rmUnwantedChars(institute.xpath('.//span/text()').extract_first())
            except IndexError or TypeError:
                orgs_info['institute'] = 'UNKNOWN'

            persLoader.add_value('organisation', orgs_info)

            # Return the individual's information
            yield persLoader.load_item()

            # Basically clicks the `next` button
            for sel in response.xpath('//a[@class="portal_navigator_next common_link"]'):
                next = sel.xpath('@href').extract_first()  # extract the link to the next page
                # print('next:', next)
                yield scrapy.Request(url=next, callback=self.parsePerson)



# =====================================================================================
def main():
    depth = 100
    delay = 0.2

    process = CrawlerProcess({
        #  'USER_AGENT': 'Chrome/58.0',
        'DEPTH_LIMIT': depth,
        'DOWNLOAD_DELAY': delay,
        'CONCURRENT_REQUESTS': 1000
    })

    spider = ResearchExplorerSpider()
    process.crawl(spider)
    process.start()  # the script will block here until the crawling is finished
    # spider.finalize()


if __name__ == "__main__":
    main()
