from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.output_parsers import ResponseSchema
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from constants import *
from gather_info_whereigo import persist_directory, embeddings

llm = ChatOllama(model="llama3.1", temperature=0.01)

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": N_DOCUMENTS})

response_schemas = [
    ResponseSchema(name="name", description="name of the location asked by the user"),
]

template = """You will receive a list of locations together with an id associated to a given location.
Also, you will receive as input from the user a given day with a time slot described by the type of activity the user wants to do in the given time slot and an historical monument.
You will have to select a location, among the ones you are going to receive, based on the activity requested and on the distance from the historical monument.
The output MUST be only a single id, nothing more.

{context}

Question: {question}"""


def get_answer(q):
    QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=[
        'context',
        'question',
    ])
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None)
    prompt = PromptTemplate(
        input_variables=["page_content", "id"],
        template="Context:\ncontent:{page_content}\nid:{id}\n",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=prompt,
        callbacks=None,
    )
    qa_chain = RetrievalQA(combine_documents_chain=combine_documents_chain,
                           retriever=retriever,
                           return_source_documents=True)

    result = qa_chain.invoke(q)

    print(result['result'])
    print("Other activities:\n")
    for doc in result['source_documents']:
        print(doc)
        print("\n")


while 1:
    # query = input("Query: ")
    get_answer("thursday 12 september 09:00 - 10:00 breakfast Mole Antoneliana")

# old messages idea: create an embedding instance of old messages and retrieve the most similar messages to the new message
