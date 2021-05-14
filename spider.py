import requests
import glob
from bs4 import BeautifulSoup
import os.path
from dateutil import parser
import pandas as pd


def download_archive_page(page):
    fname = "page-%06d.html" % page
    if not os.path.isfile(fname):
        url = (
            "https://www.reuters.com/news/archive/"
            + "?view=page&page=%d&pageSize=10" % page
        )
        r = requests.get(url)
        with open(fname, "w+") as f:
            f.write(r.text)


def parse_archive_page(page_file):
    with open(page_file, "r") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    hrefs = [
        "https://www.reuters.com" + a["href"]
        for a in soup.select("article.story div.story-content a")
    ]
    return hrefs


def download_article(url):
    # check if article already there
    fname = url.split("/")[-1] + ".html"
    if not os.path.isfile(fname):
        r = requests.get(url)
        with open(fname, "w+") as f:
            f.write(r.text)


# this needs to be updated
def parse_article(article_file):
    with open(article_file, "r") as f:
        html = f.read()
    r = {}
    soup = BeautifulSoup(html, "html.parser")
    r["id"] = soup.select_one("div.StandardArticle_inner-container")["id"]
    r["url"] = soup.find("link", {"rel": "canonical"})["href"]
    r["headline"] = soup.h1.text
    r["section"] = soup.select_one("div.ArticleHeader_channel a").text
    r["text"] = soup.select_one("div.StandardArticleBody_body").text
    r["authors"] = [
        a.text
        for a in soup.select(
            "div.BylineBar_first-container.\
                    ArticleHeader_byline-bar\
                    div.BylineBar_byline span"
        )
    ]
    r["time"] = soup.find("meta", {"property": "og:article:published_time"})["content"]
    return r


# download 10 pages of archive
for p in range(1, 10):
    download_archive_page(p)

# parse archive and add to article urls
article_urls = []

for page_file in glob.glob("page-*.html"):
    article_urls += parse_archive_page(page_file)

# download articles
for url in article_urls:
    download_article(url)

# arrange in pandas DataFrame
df = pd.DataFrame()

for article_file in glob.glob("*-id???????????.html"):
    df = df.append(parse_article(article_file), ignore_index=True)

df["time"] = pd.to_datetime(df.time)
