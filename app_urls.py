import re
import requests

from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub, Ollama
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from constants import *
from gather_info_urls import persist_directory, embeddings

llm = Ollama(model="llama3", temperature=0.1)

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": N_DOCUMENTS})

template = """You are a chatbot for a website of an italian city. You have a list of URLs of the website pages. You have to answer questions about the content of the pages. You also need to take into consideration the general knowloedge about an italian city burocracy, like the payment methods (pagoPa and bollettino postale), how to ask for an appointment and so on.

Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If the question is to request links, please only return the source links with no answer.
2. If you don't know the answer, don't try to make up an answer. Just say **I can't find the final answer but you may want to check the following links** and add the source links as a list.
3. If you find the answer, write the answer in a concise way and add the list of sources that are **directly** used to derive the answer. Exclude the sources that are irrelevant to the final answer.

{context}

Question: {question}"""


def process_llm_response(llm_response):
    answer_index = llm_response.find("Answer:")
    if answer_index != -1:
        answer_text = llm_response[answer_index + len("Answer:"):].strip()

        print(answer_text)

    sources = re.findall(r'source:(.*?)\n', llm_response, re.DOTALL)
    if sources:
        print("Sources: ")
    for source in sources:
        print("-" + source)


def get_answer(q):
    QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=[
        'context',
        'question',
    ])
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None)
    prompt = PromptTemplate(
        input_variables=["page_content", "url"],
        template="Context:\ncontent:{page_content}\nsource:{url}",
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

    print(qa_chain.invoke(q)['result'])
    # process_llm_response(qa_chain.invoke(q)['result'])


while 1:
    query = input("Query: ")
    get_answer(query)
