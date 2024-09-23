import os
import re
import sys
import numpy as np
from time import sleep
from timeit import default_timer as timer
from langchain_chroma import Chroma
from dotenv import load_dotenv
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

def RAG(question, db, rendered_tools, printScores=False, printChunks=False, printSmallestChunk=False, printQuestion=True, printContext=False, eps=35, min_samples=1, k_size=sys.maxsize, threshold=0.40) -> None:
    start = timer()    

    context = db.similarity_search_with_score(question, k=225)
    scores = [score for _, score in context]

    if (printQuestion):
        print(f"Question: {question}")

    if (printScores):
        print(f"Scores: {scores}")

    chunks = divide_into_chunks(scores, eps, min_samples)

    if (printChunks):
        print(f"Chunks: {chunks}")

    smallest_chunk = keep_smallest_chunk(chunks)

    if (printSmallestChunk):
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


    if (printContext):
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

    chain = prompt_1 | model_1 | StrOutputParser() | RunnableLambda(stripOutput)

    # User interaction 
    res = chain.invoke({"input": question}) 
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
    print(f"{final_response}\n\n")

    end = timer()
    sleep(1)
    return (end-start)

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
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

def stripOutput(received_input):
    if "I'm sorry" in received_input:
        return """{"id": 0, "name": "no_result", "arguments": {"context": [""]}}"""
    out = received_input.replace("<tool_call>","").replace("</tool_call>","").replace("\\n","").replace("'","\"").strip()
    print(out)
    return out



# Environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Find the correct context: find the chunk with the lowest distance and filter them.
persist_directory = "./chroma/expenses"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)

model_1 = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_retries=2,
)

tools = [total, average, count, print_all, no_result, select_cheapest, select_most_expensive]
rendered_tools = render_text_description(tools)

listOfQuestions1 = [
    "How much did I spend on yoga?",                                                # ok
    "Find all purchases related to 'fashion' made in September 2023.",              # ok
    "How much did I spend on clothing?",                                            # ok   
    "How much did I spend in 2023?",                                                # ok
    "How much did I spend in 2020?",                                                # ok
    "How much did I spend while hanging out?",                                      # ok, difficult to say
    "On average, how much do I spend daily?",                                       # ok
    "What was the most expensive item in the 'electronics' category?",              # ok
    "How much did the leather jacket cost?",                                        # ok
    "What is the total spent on items in the 'food' category?",                     # ok
    "List all items purchased in the 'entertainment' category.",                    # ok
    "How many electronics items were purchased after September 15th, 2023?",        # ok
    "Which purchase was made on 2023-09-21?",                                       # ok   
    "What is the price of the 'designer handbag'?",                                 # ok 
    "How many purchases were made in the 'health' category in October?",            # ok
    "What was the price of the 'fancy dinner'?",                                    # ok
    "What was bought on 2023-10-11?",                                               # ok                   
    "How many items were purchased in the 'travel' category?",                      # ok
    "Which item was the cheapest?"                                                  # ok
]

listOfQuestions2 = [
    "How many items were bought in the afternoon (12 PM - 6 PM)?",                  # not ok
    "How much was spent on 'furniture' items in total?",                            # ok
    "What is the average price of all 'electronics' items?",                        # ok
    "What item was purchased on 2023-10-01?",                                       # ok
    "What are all the items bought in the 'kitchen' category?",                     # ok
    "How many purchases were made in the 'automotive' category?",                   # ok
    "What item was bought on 2023-10-13?",                                          # ok
    "Find all items purchased on weekends (Saturday and Sunday).",                  # ok
    "What is the price of the 'new laptop'?",                                       # ok
    "What is the earliest purchase in the dataset?",                                # ok
    "Which category had the most items purchased in October?",                      # ok
    "What was the most expensive purchase in the 'health' category?",               # ok
    "How much was spent on all purchases made in October 2023?",                    # ok
    "What was purchased on 2023-09-05?",                                            # ok
    "How many items were purchased in the 'decor' category?",                       # ok
    "What was the total spent on the 'weekend trip' and 'suitcase' combined?",      # ok
    "What was the least expensive 'electronics' item?",                             # ok
    "How many purchases were made after 5 PM?",                                     # ok
    "Which item purchased in September cost exactly 100 units?",                    # ok
    "How many 'food' items were bought after September 15th, 2023?",                # ok
    "What is the price of the 'digital camera'?",                                   # ok
    "How many purchases were made in the 'furniture' category?",                    # ok
    "What was bought for 85 units?",                                                # ok
    "How much was spent on the 'tablet' and the 'gaming console' combined?",        # ok
    "What was the most expensive item purchased in the 'travel' category?",         # ok
    "Find all purchases made in the morning (before 12 PM).",                       # ok
    "How many items were bought for less than 100 units?",                          # not ok, but easy for NLP
    "What is the total amount spent on all items?",                                 # not ok, but easy for NLP
    "Which item in October was purchased for exactly 550 units?",                   # ok
    "What was the price of the 'new set of tires'?",                                # ok
    "How many items were bought for more than 300 units?",                          # not ok, but easy for NLP
    "What is the total number of purchases made in October 2023?",                  # ok
    "What item was bought for 110 units?",                                          # ok
    "How much did the 'brand new watch' cost?",                                     # ok
    "What is the price of the 'pair of wireless earbuds'?",                         # ok
    "What was the price of the 'book' bought in October 2023?"                      # ok
]

max_time = 0
total_time = 0
for question in listOfQuestions2:
    x = RAG(question, db, rendered_tools)
    total_time += x
    if x > max_time:
        max_time = x

print(f"The maximum time for a result was: {x:.2f} seconds.")
print(f"The average waiting time was: {(total_time/len(listOfQuestions2)):.2f} seconds.")