import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

@tool
def prices_sum(partial_sum: int, second_int: int) -> int:
    """Sum two integers toghether. Used to calculate the sum of the prices"""
    return partial_sum * second_int

@tool
def count_entries(partial_sum: int) -> int:
    """Adds 1 to the partial_sum for each row found"""
    return partial_sum + 1

# Environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define user question
# question = input("Question: ")
question = "How much I spent in purses?"

# Vectorial search for context
persist_directory = "./chroma/expenses"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 20})
response = retriever.invoke(question)
context = ''.join([s.page_content + '; ' for s in response])

# Define chain 
tools = [prices_sum, count_entries]
rendered_tools = render_text_description(tools)
system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:
                    {rendered_tools}
                    The context you can take your information is:
                    {context}
                    Given the user input, return the function name and arguments of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys.
                    REPLY ONLY WITH CORRECT JSON in the format {{"name": "toolname", "arguments: {{"arg1": int, "arg2":int}}}}"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

model = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_retries=2,
)

chain = prompt | model | JsonOutputParser() | tool_chain

# User interaction 
print(context)
res = chain.invoke({"input": question})
print(res)

# db_to_text()
