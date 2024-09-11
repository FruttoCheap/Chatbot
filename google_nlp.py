from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
import sqlite3

con = sqlite3.connect("googleDb.sqlite3")
cur = con.cursor()

model = Ollama(
    model="google_nlp",
    keep_alive=-1,
)

prompt = PromptTemplate.from_template("{question}")

chain = prompt | model

if __name__ == "__main__":
    question = "Show me all the expenses of september for electronics."
    resp = chain.invoke({"question": question})
    if resp.startswith("INSERT"):
        cur.execute(resp)
        con.commit()
        print("Expense added successfully")
    else:
        cur.execute(resp)
        rows = cur.fetchall()
        for row in rows:
            print(row)

