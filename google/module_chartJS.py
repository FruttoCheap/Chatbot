import json
import re
import ast
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
                                             You will output only the labels for the chart, not the corresponding values.
                                             Example UserInput: "What is the distribution of expenses by category? Give me the chart."
                                             Example Output: ["Food", "Transport", "Entertainment"]
                                             Give out the output, not the UserInput.
                                             Don't create new output, get it from {result_search}.
                                             UserInput: {question}.
                                             Type of chart: {chart_type}.
                                             Data: {result_search}.""")
    
    get_labels_chain = prompt | llm | StrOutputParser()
    return get_labels_chain

def get_label_chain():
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    prompt = PromptTemplate.from_template("""You are an expert at data visualization. You will receive a question from the user.
                                            You will understand what the user wants to know and output the label of the y axis.
                                            
                                            Example 1:
                                            UserInput: "What is the distribution of expenses by category? Give me the chart."
                                            Output: "Total expense"
                                            
                                            Example 2:
                                            UserInput: "Give me a chart that shows how many expenses I did in 2023."
                                            Output: "Number of expenses"
                                            UserInput: {question}.""")
    
    label_chain = prompt | llm | StrOutputParser()
    return label_chain

def get_data_chain():
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    prompt_template = """You are responsible for selecting the correct tool from the following list and specifying the function name and arguments based on the user's question.
                     Choose the correct tool from the list below, finding the one relevant to the question, and provide the function name and arguments to extract the relevant information from the context.
                     The number of elements in the lists must be equal to the number of labels.
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
                     Example output: {{
                         "id": 0,
                         "name": "toolname",
                         "arguments": {{
                             "context": ["row1, row2, row3"]
                         }}
                     }}
                     The argument "context" should always be a list. The rows are divided by a comma.
                     UserInput: {question}.
                     Type of chart: {chart_type}.
                     Labels: {labels}.
                     Context: {context}.
                     Get a value for each label."""


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

    data_chain = prompt | llm | StrOutputParser()  | JsonOutputParser() | tool_chain
    return data_chain


def get_graph_type(type_chain, question, PRINT_SETTINGS):
    graph_type = type_chain.invoke({"question": question})
    if (PRINT_SETTINGS["print_plot_type"]):
        print(f"Graph type: {graph_type}")
    return graph_type 

def get_labels(lables_chain, question, chart_type, result_search, PRINT_SETTINGS):
    labels = lables_chain.invoke({"question": question, "chart_type": chart_type, "result_search": result_search})
    
    if (PRINT_SETTINGS["print_plot_labels"]):
        print(f"Labels: {labels}")
    
    # Check if the input is a list-like string
    if labels.startswith("[") and labels.endswith("]"):
        # Try to safely evaluate the list-like string
        try:
            labels = ast.literal_eval(labels)
            if (PRINT_SETTINGS["print_plot_labels"]):
                print(f"Labels: {labels}")
            return labels
        except (ValueError, SyntaxError):
            # Handle invalid list-like strings
            labels = []
            if (PRINT_SETTINGS["print_plot_labels"]):
                print(f"Labels: {labels}")
            return labels
    else:
        # Handle a comma-separated string
        labels = [label.strip() for label in labels.split(',')]
        if (PRINT_SETTINGS["print_plot_labels"]):
            print(f"Labels: {labels}")
        return labels

def get_data_RAG(data_chain, question, chart_type, labels, context, PRINT_SETTINGS):
    data = data_chain.invoke({"question": question, "chart_type": chart_type, "labels": labels, "context": context, "rendered_tools": get_rendered_tools()})
    if (PRINT_SETTINGS["print_plot_data"]):
        print(f"Data: {data}")

    # Check if the input is a list-like string
    if isinstance(data, str) and data.startswith("[") and data.endswith("]"):
        # Try to safely evaluate the list-like string
        try:
            data = ast.literal_eval(data)
        except (ValueError, SyntaxError):
            # Handle invalid list-like strings
            data = []
    else:
        # Handle a comma-separated string
        data = str(data)
        data = [info.strip() for info in data.split(',')]

    if (len(data) > 1):    
        # Create a dictionary from the data
        data_dict = {data[i]: data[i+1] for i in range(0, len(data), 2)}

        # Create an aligned list for the data based on the labels
        aligned_data = [data_dict.get(label, 0.0) for label in labels]
        # Print the aligned data
        return aligned_data

    else:
        return data

def get_data_NLP(labels, context, PRINT_SETTINGS):
    # Check if the input is a list-like string
    if context.startswith("[") and context.endswith("]"):
        # Try to safely evaluate the list-like string
        try:
            context = ast.literal_eval(context)
        except (ValueError, SyntaxError):
            # Handle invalid list-like strings
            context = []
    else:
        # Handle a comma-separated string
        context = [data.strip() for data in context.split(',')]

    if (PRINT_SETTINGS["print_context"]):
        print(f"Context: {context}")

    if isinstance(context[0], str) and 'T' in context[0]:  # Check for timestamp format (with 'T')
        # Data contains dates and values
        data_dict = {entry[:10].strip('"'): context[i+1] for i, entry in enumerate(context) if i % 2 == 0}
    else:
        # Data contains simple labels and values
        data_dict = {context[i].strip('"'): context[i+1] for i in range(0, len(context), 2)}

    # Create an aligned list for the data based on the labels
    aligned_data = [data_dict.get(label.strip('"').lower(), 0.0) for label in labels]

    if (PRINT_SETTINGS["print_plot_data"]):
        print(f"Aligned Data: {aligned_data}")
    return aligned_data



def get_label_title(question, model):
    return model.invoke({"question": question})

def write_chart_html(chart_type, labels, data, label, filename="chart.html"):
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
        <title>Chart</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div style="width: 75%; margin: 0 auto;">
            <canvas id="myChart"></canvas>
        </div>

        <script>
            // Data and config from Python
            const data = {data};
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