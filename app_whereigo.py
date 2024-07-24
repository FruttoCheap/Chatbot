from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from constants import *
from gather_info_whereigo import persist_directory, embeddings

llm = Ollama(model="phi3", temperature=0.1)

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": N_DOCUMENTS})

template = """You are gonna receive as input a time slot and a day, together with a tag that describes the said time slot.
You got to return the activity that you consider the most suited for that time slot and day, among the ones that follow:

{context}

Question: {question}"""


def get_answer(q):
    # save embedding of the question
    QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=[
        'context',
        'question',
    ])
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None)
    prompt = PromptTemplate(
        input_variables=["page_content"],
        template="Context:\ncontent:{page_content}",
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

    print(qa_chain.invoke(q))
    # process_llm_response(qa_chain.invoke(q)['result'])


while 1:
    query = input("Query: ")
    get_answer(query)

# old messages idea: create an embedding instance of old messages and retrieve the most similar messages to the new message
