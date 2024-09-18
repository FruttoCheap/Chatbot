from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_text_splitters import CharacterTextSplitter
from pydantic.v1 import BaseModel, Field
import shutil
import os

def clear_chroma(persist_directory):
    """Delete the Chroma directory to clear all existing data."""
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

persist_directory = "./chroma/expenses"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)

model = OllamaFunctions(
    model="google_rag",
    keep_alive=-1,
    format="json",
)

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 5})

template = """Context: {context}. Question: {question}"""

def db_to_text():
    import sqlite3
    clear_chroma(persist_directory)  # Clear existing data
    con = sqlite3.connect("googleDb.sqlite3")
    cur = con.cursor()

    cur.execute("SELECT * FROM expensesok;")
    rows = cur.fetchall()

    for row in rows:
        # Adjust date format to include time
        date_time_str = row[3]  # Assuming row[3] contains the full datetime string
        date_time_obj = datetime.strptime(date_time_str, "%Y-%m-%dT%H:%M:%S")

        # Extract day, month, year, and hour
        day = date_time_obj.day
        month = date_time_obj.strftime("%B")
        year = date_time_obj.year
        hour = date_time_obj.strftime("%H:%M:%S")  # Format hour and minute

        if 4 <= day <= 20 or 24 <= day <= 30:
            suffix = "th"
        else:
            suffix = ["st", "nd", "rd"][day % 10 - 1]

        ordinal_day = f"{day}{suffix}"

        # Include hour in the text
        text = f"Price: {row[0]}, Description: {row[1]}, Date: {ordinal_day} {month} {year} {hour}"
        embed(row[1], text)

def embed(description, text):
    texts = text_splitter.split_text(text)
    if texts:
        Chroma.from_texts([t for t in texts], embeddings, persist_directory=persist_directory,
                          metadatas=[{"row": description}])

def get_chain(prompt):

    class Output(BaseModel):
        """Select the expense/expenses requested by the user and provide the price/total and description/summary of the selected expense/expenses."""

        total: float = Field(description="The price of the selected expense or the sum of the prices of the selected expenses.")
        description: str = Field(description="The description of the selected expense or a summary of the selected expenses.")

    return prompt | model.with_structured_output(Output)

def get_context(q):
    response = retriever.invoke(q)
    return ''.join([s.page_content + '; ' for s in response])

def get_answer(q):
    prompt = PromptTemplate.from_template(template)
    chain = get_chain(prompt)
    context = get_context(q)
    try:
        return chain.invoke({"question": q, "context": context})
    except Exception as e:
        print(e)
        return "I'm sorry, I couldn't find any expenses matching your request."

# Run the data import and processing
# db_to_text()

# Example query
# query = "How much did I spend on dining?"
# print(get_answer(query))