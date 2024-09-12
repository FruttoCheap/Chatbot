from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_text_splitters import CharacterTextSplitter
from pydantic.v1 import BaseModel, Field


class Expense(BaseModel):
    """Outputs the structured version of the user input."""

    price: float = Field(description="The sum of money spent by the user")
    description: str = Field(description="A short description of the user's expense")


class Output(BaseModel):
    """Outputs the list of expenses coherent with the user's input, taken from the list inside the input."""

    correct_expenses: list[Expense] = Field(description="The list of expenses coherent with the user's input")


persist_directory = "./chroma/expenses"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

model = OllamaFunctions(
    model="llama3.1",
    keep_alive=-1,
    format="json",
)

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 20})

template = """Context: {context}. Question: {question}"""


def get_chain(prompt):
    model_with_tools = model.with_structured_output(Output)
    return prompt | model_with_tools


def get_context(q):
    response = retriever.invoke(q)
    return ''.join([s.page_content + '; ' for s in response])


def get_answer(q):
    chain = get_chain(PromptTemplate.from_template(template))
    context = get_context(q)
    try:
        return chain.invoke({"context": context, "question": q})
    except Exception as e:
        print(e)
        return "I'm sorry, I couldn't find any expenses matching your request."


while 1:
    query = input("Query: ")
    print(get_answer(query))


# db_to_text()
