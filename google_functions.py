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


@tool
def sum_prices(prices: list[float]) -> float:
    """Sums the list of prices obtained from the context."""
    total = 0
    for price in prices:
        total += price
    return total


tools = [sum_prices]

persist_directory = "./chroma/expenses"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

model = OllamaFunctions(
    model="llama3.1",
    keep_alive=-1,
    format="json",
)

db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 50})

template = """Context: {context}. Question: {question}"""


def get_prompt():
    examples = [
        HumanMessage(
            "Context: Price: 100.0, Description: new pair of shoes, Date: 2023-09-01, Price: 95.0; Description: pair of running shoes, Date: 2023-11-28; Question: How much did I spend on shoes?", name="example_user"
        ),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "sum_prices", "args": {"prices": [95.0, 100.0]}, "id": "1"}
            ],
        ),
        ToolMessage("195.0", tool_call_id="1"),
        AIMessage(
            "total: 195.0",
            name="example_assistant",
        ),
    ]

    system = """You are bad at math but are an expert at using a calculator. 
    You will receive a list of expenses and must calculate the total sum of the expenses, which must be selected based on the user input.

    Use past tool usage as an example of how to correctly use the tools."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            *examples,
            ("human", "Expenses: {context}. Question: {question}"),
        ]
    )


def get_chain(prompt):
    model_with_tools = model.bind_tools(tools)
    return model_with_tools

    # return {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | model_with_tools


def get_context(q):
    response = retriever.invoke(q)
    return ''.join([s.page_content + '; ' for s in response])


def get_answer(q):
    chain = get_chain(get_prompt())
    context = get_context(q)
    messages = [HumanMessage(f"Context: {context} Question: {q}")]
    try:
        ai_msg = chain.invoke(messages)
        messages.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            selected_tool = {"sum_prices": sum_prices}[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            messages.append(tool_msg)

        print(messages)
        return chain.invoke(messages)

    except Exception as e:
        print(e)
        return "I'm sorry, I couldn't find any expenses matching your request."


while 1:
    query = input("Query: ")
    print(get_answer(query))


# db_to_text()
