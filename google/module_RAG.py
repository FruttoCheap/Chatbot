import re
import sys
import numpy as np
from timeit import default_timer as timer
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_groq import ChatGroq
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda 
from sklearn.cluster import DBSCAN

@tool
def total(context: list) -> int:
    """Extracts the prices and returns the total expense value. 'context' is the list of all expenses."""
    prices = []
    
    for result in context:
        # Correct regex pattern to match floating-point numbers
        x = re.findall(r"\d+\.\d+", str(result))
        if x:
            for i in x:
                prices.append(float(i))

    return sum(prices)
@tool
def average(context: list) -> int:
    """Returns the average expense value. 'context' is the list of all results. Instead of dividing by zero, returns a test sentence."""
    prices = []
    
    if (len(context) != 0):
        for result in context:
            x = re.findall(r"\d+\.\d", result)
            if x:
                for i in x:
                    prices.append(float(i))
    else:
        return "The RAG was not sufficient. Try another approach."

    avg = round(float(sum(prices) / len(prices)), 2) if len(prices) != 0 else 0

    return avg
@tool
def count(context: list) -> int:
    """Returns the number of rows that are relevant to the question."""
    return len(context)
@tool
def print_all(context: list) -> str:
    """Returns all the rows that are relevant to the question."""
    return '\n'.join(map(str, context))
@tool
def no_result(context: list) -> str:
    """Returns a message if there are no results."""
    return "RAG was not sufficient. Try with NLP."
@tool
def select_cheapest(context: list) -> str:
    """Returns the cheapest item from the context."""
    prices = []
    items = []
    
    for result in context:
        x = re.findall(r"\d+\.\d+", str(result))
        if x:
            for i in x:
                prices.append(float(i))
                items.append(result)

    min_price = min(prices)
    index = prices.index(min_price)
    return items[index]
@tool
def select_most_expensive(context: list) -> str:
    """Returns the most expensive item from the context."""
    prices = []
    items = []
    
    for result in context:
        x = re.findall(r"\d+\.\d+", str(result))
        if x:
            for i in x:
                prices.append(float(i))
                items.append(result)

    max_price = max(prices)
    index = prices.index(max_price)
    return items[index]


def divide_into_chunks(numbers, eps=1.0, min_samples=2):
    # Convert numbers to a 2D array for DBSCAN
    numbers_array = np.array(numbers).reshape(-1, 1)
    
    # Perform DBSCAN clusteringX
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(numbers_array)
    labels = dbscan.labels_
    
    # Create chunks based on cluster labels
    chunks = {}
    for num, label in zip(numbers, labels):
        if label not in chunks:
            chunks[label] = []
        chunks[label].append(num)
    
    # Filter out noise (label -1 is used for noise by DBSCAN)
    chunks = {label: chunk for label, chunk in chunks.items() if label != -1}
    
    return chunks

def keep_smallest_chunk(chunks):
    # Find the chunk with the minimum values
    min_chunk = min(chunks.values(), key=lambda chunk: np.mean(chunk))
    return min_chunk

def tool_chain(model_output):
    tools = get_tools() 
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

def get_tools():
    return [total, average, count, print_all, no_result, select_cheapest, select_most_expensive]

def get_rendered_tools():
    tools = get_tools()
    rendered_tools = render_text_description(tools)
    return rendered_tools

def stripOutput(received_input):
    if "I'm sorry" in received_input:
        return """{"id": 0, "name": "no_result", "arguments": {"context": [""]}}"""
    out = received_input.replace("<tool_call>","").replace("</tool_call>","").replace("\\n","").replace("'","\"").strip()
    return out

def RAG(question, db, stripOutput, PRINT_SETTINGS, k_size, eps=35, min_samples=1, threshold=0.40) -> None:
    start = timer()    
    context = db.similarity_search_with_score(question, k=k_size)
    scores = [score for _, score in context]

    if PRINT_SETTINGS["print_question"]:
        print(f"Question: {question}")

    if PRINT_SETTINGS["print_scores"]:
        print(f"Scores: {scores}")

    chunks = divide_into_chunks(scores, eps, min_samples)

    if PRINT_SETTINGS["print_chunks"]:
        print(f"Chunks: {chunks}")

    smallest_chunk = keep_smallest_chunk(chunks)

    if PRINT_SETTINGS["print_smallest_chunk"]:
        print(f"Smallest Chunk: {smallest_chunk}")


    # Calculate the threshold score
    max_score = max(smallest_chunk)
    threshold_score = max_score * threshold - 100

    # Filter context based on the threshold score
    context = [[(doc, score) for doc, score in context if score in smallest_chunk and score <= min(smallest_chunk) + 75 and score >= threshold_score]]
    context_new = []
    cnt = 0
    for i in context:
        for j in i:
            for k in j:
                if cnt % 2 == 0:
                    context_new.append(f"{k.page_content}\n")
                cnt += 1


    if PRINT_SETTINGS["print_context"]:
        for i in context_new:
            print(i, end="")

    system_prompt = f"""
                        You are an expert extraction algorithm. Your goal is to extract the relevant information from the context to answer the user's question.
                        Only extract relevant information from the text. Do not add any new information.
                        This is the context:
                        {context_new}. 
                        Give back the information I should use from {context_new} to answer.
                        Select the information from {context_new}. Not necessarely all of it: if you think a row is not relevant, ignore it.
                        Put all of these information into a Python list, and give me just that.
                        PUT INTO THE LIST ALL RELEVANT INFORMATION FROM THE CONTEXT.
                        Do a stricter selection.
                        Don't print anything that is not in the context.
                        Possible outputs: [list of row and purchases] or [empty list]
                        Return the empty list if you think there is no relevant information.

                        IF YOU THINK YOU CAN'T PERFORM THE TASK, RETURN AN EMPTY LIST.
                        Do not return any information that is not in the context. Just select the row. Don't select only numbers.
                        - Return list of strings
                        - Return list of strings
                        - Return list of strings
                        - Return list of strings
                        """
    
    prompt_1 = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    model_1 = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0, max_retries=2)
    chain = prompt_1 | model_1 | StrOutputParser() | RunnableLambda(stripOutput)

    # User interaction 
    res = chain.invoke({"input": question})

    rendered_tools = get_rendered_tools() 
    # Define chain 
    system_prompt_tools = f"""
                            You are responsible for selecting the correct tool from the following list and specifying the function name and arguments based on the user's question.
                            Choose the correct tool from the list below, finding the one relevant to the question, and provide the function name and arguments to extract the relevant information from the context.
                            Here are the tools available and their descriptions:
                            {rendered_tools}
                            The input you have access to includes:
                            {res}
                            Your goal:
                            - Select the correct tool for the task.
                            - Ensure **every** part of the context is passed to the tools.
                            - Provide a response in JSON format with 'name' and 'arguments' keys.
                            - If you believe you don't have enough context, use the no_result tool.
                            - If a day, month, year or time is mentioned in the input, use the no_result tool.
                            - If a day, month, year or time is mentioned in the input, use the no_result tool.
                            - If a day, month, year or time is mentioned in the input, use the no_result tool.
                            - If the {res} is empty, use the no_result tool.
                            - If the {res} is empty, use the no_result tool.
                            - If the {res} is empty, use the no_result tool.
                            - If the {res} is empty, use the no_result tool.
                            - If "average" or "mean" in {res}, use the average tool.
                            """
    
    model_with_tools = ChatGroq(
        model="llama3-groq-70b-8192-tool-use-preview",
        temperature=0
    )

    prompt_2 = ChatPromptTemplate.from_messages(
        [("system", system_prompt_tools), ("user", "{input}")]
    )
    final_chain = prompt_2 | model_with_tools | StrOutputParser() | RunnableLambda(stripOutput) | JsonOutputParser() | tool_chain
    final_response = final_chain.invoke({"input": question})

    end = timer()
    if PRINT_SETTINGS["print_time"]:
        print(f"Time: {end-start}")
    
    return final_response

def get_embedded_database(persist_directory):
    persist_directory = "./chroma/expenses"
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings)

    return db

def get_RAG_model():
    model_1 = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0, max_retries=2)
    return model_1

