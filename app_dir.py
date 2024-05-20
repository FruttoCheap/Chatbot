from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from constants import *
from gather_info_dir import persist_directory, embeddings

llm = Ollama(model="llama3", temperature=0.1)

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": N_DOCUMENTS})

template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If the question is to request references, please only return the source with no answer.
2. If you don't know the answer, don't try to make up an answer. Just say **I can't find the final answer but you may want to check the following references** and add the sources as a list.
3. If you find the answer, write the answer in a concise way and add the list of sources that are **directly** used to derive the answer. Exclude the sources that are irrelevant to the final answer.

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

    print(qa_chain.invoke(q)['result'])
    # process_llm_response(qa_chain.invoke(q)['result'])


while 1:
    query = input("Query: ")
    get_answer(query)

# old messages idea: create an embedding instance of old messages and retrieve the most similar messages to the new message
