import sqlite3

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

website = ""
urls = []

db_name = "urls.db"

con = sqlite3.connect(db_name)
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS urls (url)")
con.commit()

persist_directory = "./chroma"

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

embeddings = HuggingFaceEmbeddings()

blacklist = [
    '[document]',
    'noscript',
    'header',
    'html',
    'meta',
    'head',
    'input',
    'script',
    'style'
    # name more elements if not required
]


def find_sub_pages(main_url):
    res = requests.get(main_url)
    html_page = res.content
    try:
        soup = BeautifulSoup(html_page, 'html.parser')
    except:
        return
    for link in soup.find_all('a', href=True):
        url = str(link.get('href'))
        if url[0] == "/":
            url = website + url[1:]
            # if url not in urls and url != website and 'file' not in url:
            if url not in urls and url != website and "file" not in url:
                urls.append(url)
                cur.execute("INSERT INTO urls VALUES (?)", (url,))
                con.commit()
                find_sub_pages(url)


def get_text_from_url(url):
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(string=True)
    page = url + '\n'
    for t in text:
        if t.parent.name not in blacklist and t != 'Immagine':
            page += '{} '.format(t)

    return page


def embed():
    for u in cur.execute("SELECT * FROM urls"):
        text = get_text_from_url(u[0]).replace("\n\n", "\n")
        texts = text_splitter.split_text(text)
        vectordb = Chroma.from_texts(texts, embeddings, persist_directory=persist_directory,
                                     metadatas=[{"url": u[0]}, ])
        vectordb.persist()


if __name__ == '__main__':
    website = input("Enter website: ")
    if website != "":
        find_sub_pages(website)
        embed()
