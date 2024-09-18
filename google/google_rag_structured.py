import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import tool

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

# class Expense(BaseModel):
#     """Outputs the structured version of the user input."""

#     price: float = Field(description="The sum of money spent by the user")
#     description: str = Field(description="A short description of the user's expense")


# class Output(BaseModel):
#     """Outputs the list of expenses coherent with the user's input, taken from the list inside the input."""

#     correct_expenses: list[Expense] = Field(description="The list of expenses coherent with the user's input")


load_dotenv()

persist_directory = "./chroma/expenses"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 20})

template = """Context: {context}. Question: {question}"""

# model = OllamaFunctions(
#     model="llama3.1",
#     keep_alive=-1,
#     format="json",
# )

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_retries=2,
)

# my_account_id = os.getenv("CF_ACCOUNT_ID")
# my_api_token = os.getenv("CF_API_KEY")
# model = CloudflareWorkersAI(account_id=my_account_id, api_token=my_api_token)


def get_chain(prompt):
    model_with_tools = model.with_structured_output(Output)
    return prompt | model_with_tools


def get_context(q):
    response = retriever.invoke(q)
    return ''.join([s.page_content + '; ' for s in response])


def get_answer(q):
    chain = get_chain(PromptTemplate.from_template(template))
    context = get_context(q)
    print(context)
    try:
        return chain.invoke({"context": context, "question": q})
    except Exception as e:
        print(e)
        return "I'm sorry, I couldn't find any expenses matching your request."


while 1:
    query = input("Query: ")
    answer = get_answer(query)
    total = 0
    for expense in answer.correct_expenses:
        print(expense)
        total += expense.price
    print(f"Total: {total}")


# db_to_text()
