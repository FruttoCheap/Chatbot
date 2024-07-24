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
    cur.execute("SELECT * FROM itinerary_location")
    for u in cur.fetchall():
        cur.execute(f"SELECT * FROM itinerary_openhours WHERE id = {u[11]}")
        opening_hours = cur.fetchall()
        opening_hours_text = ", Opening hours: Monday: " + opening_hours[0][1] + ", Tuesday: " + opening_hours[0][2] + ", Wednesday: " + opening_hours[0][3] + ", Thursday: " + opening_hours[0][4] + ", Friday: " + opening_hours[0][5] + ", Saturday: " + opening_hours[0][6] + ", Sunday: " + opening_hours[0][7]
        text = "Title: " + u[1] + ", Rating: " + str(u[2]) + ", Price: " + u[4] + ", Type: " + u[5] + ", Description: " + u[9] + opening_hours_text
        texts = text_splitter.split_text(text)
        vectordb = Chroma.from_texts(texts, embeddings, persist_directory=persist_directory,
                                     metadatas=[{"id": u[0]}, ])
        vectordb.persist()


if __name__ == '__main__':
    embed()
    con.close()
