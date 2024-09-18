import os
from dotenv import load_dotenv
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

def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

def stripOutput(received_input):
    return received_input.replace("<tool_call>","").replace("</tool_call>","").replace("\\n","").strip()

@tool
def prices_sum(prices: list) -> int:
    """Insert into the prices list of only the relevant purchases and returns the sum."""
    return sum(prices)

@tool
def count_entries(partial_sum: int) -> int:
    """Adds 1 to the partial_sum for each row found"""
    return partial_sum + 1

# Environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define user question
# question = input("Question: ")
question = "How much I spent in handbags?"

# Vectorial search for context
persist_directory = "./chroma/expenses"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory=persist_directory,
            embedding_function=embeddings,
             collection_metadata={"hnsw:space": "cosine"})

retriever = db.as_retriever(search_type="similarity_score_threshold", 
                                 search_kwargs={"score_threshold": 0.9}, )
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
                 """
prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

model = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_retries=2,
)

chain = prompt | model | StrOutputParser() | RunnableLambda(stripOutput)


# User interaction 
res = chain.invoke({"input": question})
print(res) | JsonOutputParser() | tool_chain

# db_to_text()
