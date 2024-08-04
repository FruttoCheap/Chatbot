import sqlite3

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

db_name = "whereigo.sqlite3"

con = sqlite3.connect(db_name)
cur = con.cursor()

persist_directory = "./chroma"

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")


def embed():
    cur.execute("SELECT * FROM itinerary_mustsee")
    for u in cur.fetchall():
        print(u)
        text = "Title: " + u[1] + ", Rating: " + str(u[4]) + ", Description: " + u[3]
        texts = text_splitter.split_text(text)
        print(texts)
        vectordb = Chroma.from_texts(texts, embeddings, persist_directory=persist_directory,
                                     metadatas=[{"id": u[0]}, ])
        vectordb.persist()


if __name__ == '__main__':
    embed()
    con.close()
