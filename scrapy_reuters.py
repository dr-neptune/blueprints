import scrapy
from scrapy import Spider
import logging


class ReutersArchiveSpider(scrapy.Spider):
    name = "reuters-archive"

    custom_settings = {
        "LOG_LEVEL": logging.WARNING,
        "FEED_FORMAT": "json",
        "FEED_URI": "reuters-archive.json",
    }

    start_urls = ["https://www.reuters.com/news/archive"]

    def parse(self, response):
        for article in response.css("article.story div.story-content a"):
            yield response.follow(
                article.css("a::attr(href)").extract_first(), self.parse_article
            )

        next_page_url = response.css("a control-nav-next::attr(href)").extract_first()
        if (next_page_url is not None) and ("page=2" not in next_page_url):
            yield response.follow(next_page_url, self.parse)

    def parse_article(self, response):
        yield {"title": response.css("h1::text").extract_first().strip()}


from scrapy.crawler import CrawlerProcess

process = CrawlerProcess()

process.crawl(ReutersArchiveSpider)
process.start()
