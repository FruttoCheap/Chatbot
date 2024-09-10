from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from pydantic.v1 import BaseModel, Field

from constants import *
from gather_info_dir import persist_directory, embeddings


class LocationDetail(BaseModel):
    """Select the best machine based on the requested characteristics"""

    name: str = Field(description="The name of the selected machine")
    characteristics: str = Field(description="The characteristics of the selected machine")
    production_requirements: str = Field(description="The production requirements of the selected machine")


model = OllamaFunctions(
    model="llama3",
    keep_alive=-1,
    format="json",
    temperature=0.1,
)

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": N_DOCUMENTS})

template = """You are a chatbot to help choosing the right machine for putting caps on bottles, based on the characteristics of the machine and the production requirements.
These are the characteristics of the machines:

{context}

The question you have to answer is:

{question}

Yiu MUST adhere to the following schema:
MachineInput:
    string name;
    string characteristics;
    string production_requirements;
Output:
    string name;"""


def get_chain(prompt):
    structured_llm = model.with_structured_output(LocationDetail)
    return prompt | structured_llm


def get_context(q):
    response = retriever.invoke(q, k=5)
    print(response)
    return response


def get_answer(q):
    prompt = PromptTemplate.from_template(template)
    chain = get_chain(prompt)
    context = get_context(q)
    result = chain.invoke({"question": q, "context": context})
    return result.name


while 1:
    query = input("Query: ")
    get_answer(query)

# old messages idea: create an embedding instance of old messages and retrieve the most similar messages to the new message
