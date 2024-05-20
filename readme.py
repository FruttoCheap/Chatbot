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

template = """Use the following Template as a template for a readme file. Please fill in the Template using the information gathered in the Context section.

Template:

# Project Title

A brief description of what this project does and who it's for.

## Installation

Describe the installation process here.

```bash
pip install your_project

Context:

{context}

Question: {question}"""


def get_answer(q):
    QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=[
        'context',
        'question',
    ])
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None)
    prompt = PromptTemplate(
        input_variables=["page_content"],
        template="Context:\ncontent:{page_content}\nsource:{file}",
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
    query = "main functionality"
    get_answer(query)

# old messages idea: create an embedding instance of old messages and retrieve the most similar messages to the new message
