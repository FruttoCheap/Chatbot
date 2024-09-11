import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import sqlite3

con = sqlite3.connect("googleDb.sqlite3")
cur = con.cursor()

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model = OllamaFunctions(
    model="llama3",
    keep_alive=-1,
    format="json",
    temperature=0.1,
)

prompt = PromptTemplate.from_template("")

chain = prompt | model

if __name__ == "__main__":

