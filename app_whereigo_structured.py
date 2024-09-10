from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from openai import OpenAI

from constants import *
from gather_info_whereigo import persist_directory, embeddings

from pydantic import BaseModel, Field
import instructor

from gather_info_whereigo_position import get_locations


class LocationDetail(BaseModel):
    """Select the best location based on the user's preferences"""

    id: int = Field(description="The id of the selected location")


# client = instructor.from_openai(
#     OpenAI(
#         base_url="http://localhost:11434/v1",
#         api_key="ollama",  # required, but unused
#     ),
#     mode=instructor.Mode.JSON,
# )

model = OllamaFunctions(
    model="location",
    keep_alive=-1,
    format="json",
)

systemPrompt = """You are a programmatic location recommender.
Each location that you receive in the Context MUST adhere to the following schema: 
id: int, name: string, rating: string, price: string, description: string, opening_hours: string, attributes: string, distance: float.
Validate them and throw an error if they are not valid.
Also, you will receive a UserInput, that MUST follow the following schema: 
date: string, time_slot: string, tag: string, preferences: string.
You must return the location id of the location that best fits the user's preferences and desired time slot 
and tag (ex. breakfast, lunch, disco, etc...), chosen among the ones in the context.
"""

# db = Chroma(persist_directory=persist_directory,
#             embedding_function=embeddings)
#
# retriever = db.as_retriever(search_kwargs={"k": N_DOCUMENTS})

# Used to obtain the locations from the embedded database, now using position
# def get_locations(query):
#     documents = retriever.invoke(query)
#     response = ""
#     for doc in documents:
#         response += f"id: {doc.metadata["id"]} of location: {doc.page_content}\n"
#
#     return response


def get_prompt():
    template = """UserInput: {question}.

Context:
{context}."""
    prompt = PromptTemplate.from_template(template)
    return prompt


# def get_answer(t):
#     response = client.chat.completions.create(
#         model="llama3.1",
#         messages=[
#             {
#                 "role": "system",
#                 "content": systemPrompt,
#             },
#             {
#                 "role": "user",
#                 "content": t,
#             }
#         ],
#         temperature=0.01,
#         response_model=LocationDetail,
#     )
#     print(response.id)


def get_chain(prompt):
    structured_llm = model.with_structured_output(LocationDetail)
    return prompt | structured_llm


def get_location_id(date, time_slot, tag, preferences, lat, lon):
    # question = f"date: {date}, time_slot: {time_slot}, tag: {tag}, preferences: {preferences}"
    context = get_locations(lat, lon)
    question = "date: 2024-05-04, time_slot: 09:00:00 - 10:00:00, tag: lunch, preferences: low cost, avoid: fast food"
    chain = get_chain(get_prompt())
    result = chain.invoke({"question": question, "context": context})
    return result.id


if __name__ == '__main__':
    print(get_location_id("thursday 12 september", "13:00 - 14:00", "lunch", "oriental food", 45.0703, 7.6869))
