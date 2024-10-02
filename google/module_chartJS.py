import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda 
from langchain_core.tools import tool
from operator import itemgetter
from langchain.tools.render import render_text_description

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
    return "No match found."
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
    out = received_input.replace("<tool_call>","").replace("</tool_call>","").replace("\\n","").replace("'","\"").replace('["','').replace('"]','').replace('"', "").replace(", ", "\n").replace(",","\n").strip()
    return out

def get_type_chain():
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    prompt = PromptTemplate.from_template("""You are an expert at data visualization. You will receive a question from the user.
                                            You will have, as data, a list of expenses with the columns price, description, category, and timestamp.
                                            You will output the best type of chart to answer the question.
                                            The chart can be one of the following types: bar, pie, doughnut, line, polarArea, radar. 
                                            Example UserInput: "What is the distribution of expenses by category? Give me the chart."
                                            Example Output: "pie"
                                            UserInput: {question}.""")
    
    get_type_chain = prompt | llm | StrOutputParser()
    return get_type_chain

def get_labels_chain():
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    prompt = PromptTemplate.from_template("""You are an expert at data visualization. You will receive a question from the user, the type of chart to realize and the data to use.
                                             You will have, as data, a list of expenses with the columns price, description, category, and timestamp.
                                             You will output the labels for the chart.
                                             Example UserInput: "What is the distribution of expenses by category? Give me the chart."
                                             Example Output: ["Food", "Transport", "Entertainment"]
                                             UserInput: {question}.
                                             Type of chart: {chart_type}.
                                             Data: {result_search}.""")
    
    get_labels_chain = prompt | llm | StrOutputParser()
    return get_labels_chain

def get_data_chain():
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    prompt_template = """You are responsible for selecting the correct tool from the following list and specifying the function name and arguments based on the user's question.
                         Choose the correct tool from the list below, finding the one relevant to the question, and provide the function name and arguments to extract the relevant information from the context.
                         Here are the tools available and their descriptions:
                         {rendered_tools}
                         Your goal:
                         - Select the correct tool for the task.
                         - Ensure **every** part of the context is passed to the tools.
                         - Provide a response in JSON format with 'name' and 'arguments' keys.
                         - If you believe you don't have enough context, use the no_result tool.
                         - If a day, month, year or time is mentioned in the input, use the no_result tool.
                         - If the context is empty, use the no_result tool.
                         - If "average" or "mean" in context, use the average tool.
                         Always output a readable JSON format for the JsonOutputParser.
                         Example output: id: 0 name: total arguments: context: [250.0]"
                         The argument "context" should always be a list. The rows are divided by a comma.
                         UserInput: {question}.
                         Type of chart: {chart_type}.
                         Labels: {labels}.
                         Context: {context}."""

    # Create a prompt using the ChatPromptTemplate with variables properly mapped
    prompt = ChatPromptTemplate.from_template(
        prompt_template
    )

    # Define the input variables explicitly for the chain
    prompt = prompt.partial(
        rendered_tools="{rendered_tools}",
        question="{question}",
        chart_type="{chart_type}",
        labels="{labels}",
        context="{context}"
    )

    data_chain = prompt | llm | StrOutputParser() | RunnableLambda(stripOutput) | JsonOutputParser() | tool_chain
    return data_chain


def get_graph_type(type_chain, question):
    return type_chain.invoke({"question": question})   

def get_labels(lables_chain, question, chart_type, result_search):
    return lables_chain.invoke({"question": question, "chart_type": chart_type, "result_search": result_search})

def get_data(data_chain, question, chart_type, labels, context):
    res = data_chain.invoke({"question": question, "chart_type": chart_type, "labels": labels, "context": context, "rendered_tools": get_rendered_tools()})
    print(res)
    return res        
def write_chart_html(chart_type, labels, data, label="test", filename="chart.html"):
    data = {
        'labels': labels,
        'datasets': [{
            'label': label,
            'data': data,
            'backgroundColor': 'rgba(75, 192, 192, 0.2)',
            'borderColor': 'rgba(75, 192, 192, 1)',
            'borderWidth': 1
        }]
    }

    config = {
        'type': chart_type,  # You can change this to 'bar', 'pie', 'doughnut', 'line', 'polarArea', 'radar'.
        'data': data,
        'options': {
            'responsive': True,
            'scales': {
                'y': {
                    'beginAtZero': True
                }
            }
        }
    }

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chart.js Example</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div style="width: 75%; margin: 0 auto;">
            <canvas id="myChart"></canvas>
        </div>

        <script>
            // Data and config from Python
            const data = {json.dumps(data)};
            const config = {json.dumps(config)};

            // Render the chart
            const myChart = new Chart(
                document.getElementById('myChart'),
                config
            );
        </script>
    </body>
    </html>
    """

    with open(filename, 'w') as f:
        f.write(html_content)