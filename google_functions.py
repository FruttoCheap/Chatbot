from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
import sqlite3
from datetime import datetime


def parse_date_to_iso(date_str, date_format):
    # Parse the date string into a datetime object
    date_obj = datetime.strptime(date_str, date_format)
    # Convert the datetime object to ISO format
    return date_obj.isoformat()


con = sqlite3.connect("googleDb.sqlite3")
cur = con.cursor()

cur.execute("SELECT * FROM expenses;")
rows = cur.fetchall()
for row in rows:
    print(row[0])
    date_format = '%Y-%m-%d %H:%M:%S'
    iso_date = parse_date_to_iso(row[3], date_format)
    cur.execute(f"INSERT INTO expensesok (price, description, category, date) VALUES ('{row[0]}','{row[1]}','{row[2]}','{iso_date}');")

con.commit()

# model = Ollama(
#     model="google_nlp",
#     keep_alive=-1,
# )
#
# prompt = PromptTemplate.from_template("{question}")
#
# chain = prompt | model
#
# if __name__ == "__main__":
#     question = "How much did I spent for each category during september?"
#     resp = chain.invoke({"question": question})
#     if resp.startswith("INSERT"):
#         cur.execute(resp)
#         con.commit()
#         print("Expense added successfully")
#     else:
#         cur.execute(resp)
#         rows = cur.fetchall()
#         for row in rows:
#             print(row)

