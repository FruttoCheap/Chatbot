from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_text_splitters import CharacterTextSplitter
from pydantic.v1 import BaseModel, Field

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
    con = sqlite3.connect("googleDb.sqlite3")
    cur = con.cursor()

    cur.execute("SELECT * FROM expenses;")
    rows = cur.fetchall()

    for row in rows:
        date_obj = datetime.strptime(row[3].split(" ")[0], "%Y-%m-%d")

        # Extract day, month, and year
        day = date_obj.day
        month = date_obj.strftime("%B")
        year = date_obj.year

        if 4 <= day <= 20 or 24 <= day <= 30:
            suffix = "th"
        else:
            suffix = ["st", "nd", "rd"][day % 10 - 1]

        ordinal_day = f"{day}{suffix}"

        embed(row[1], "Price: " + str(row[0]) + ", Description: " + row[1] + ", Date: " + f"{ordinal_day} {month} {year}")


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


while 1:
    query = input("Query: ")
    print(get_answer(query))


# db_to_text()
