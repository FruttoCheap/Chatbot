from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from pydantic import BaseModel, Field

import sqlite3

from constants import *
from gather_info_whereigo_mustsee import persist_directory, embeddings


class LocationDetail(BaseModel):
    """Select the best locations based on the user's preferences"""

    ids: list[int] = Field(description="The ids of the selected locations")


model = OllamaFunctions(
    model="must_see",
    keep_alive=-1,
    format="json",
)

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": N_DOCUMENTS})

template = """Context: {context}

UserInput: {question}"""


def get_context(city):
    db_name = "whereigo.sqlite3"
    con = sqlite3.connect(db_name)
    cur = con.cursor()

    cur.execute(f"SELECT id, title, description, rating FROM itinerary_mustsee mu, itinerary_activity a WHERE mu.activity_ptr_id = a.id AND city_id = '{city}' ORDER BY rating DESC LIMIT 0, 20;")

    documents = cur.fetchall()
    context = ""
    for doc in documents:
        context += f"id: {doc[0]}, name: {doc[1]}, rating: {doc[3]}, description: {doc[2]}\n"

    return context


def get_answer(city, n, preferences):
    prompt = PromptTemplate(template=template, input_variables=[
        'context',
        'question',
    ])
    llm = model.with_structured_output(LocationDetail)
    qa_chain = prompt | llm
    # prompt = PromptTemplate(
    #     input_variables=["page_content", "id"],
    #     template="Context:\ncontent:{page_content}\nid:{id}\n",
    # )
    # combine_documents_chain = StuffDocumentsChain(
    #     llm_chain=llm_chain,
    #     document_variable_name="context",
    #     document_prompt=prompt,
    #     callbacks=None,
    # )
    # qa_chain = RetrievalQA(combine_documents_chain=combine_documents_chain,
    #                        retriever=retriever, return_source_documents=True)

    context = get_context(city)
    question = f"n: {n}, preferences: {preferences}"
    result = qa_chain.invoke({"context": context, "question": question})
    print(result.ids)
    # process_llm_response(qa_chain.invoke(q)['result'])


if __name__ == '__main__':
    get_answer("Turin", 5, "Museums")
