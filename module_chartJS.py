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
def print_all(context: list) -> str:
    """Returns all the rows that are relevant to the question."""
    return '\n'.join(map(str, context))

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
def count(context: list) -> float:
    """Returns the number of rows that are relevant to the question."""
    return len(context)
@tool
def get_price(context: list) -> list:
    """Extracts the price from the context."""
    prices = []
    
    for result in context:
        if result.isdigit():
            prices.append(float(result))

    return prices
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
    return [total, average, count, get_price, print_all, select_cheapest, select_most_expensive]

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
                                             If you have labels to be time, order them in chronological order. Do not translate dates in English.
                                             If you have a time indication, do not ignore it. Questions like "through the years" should have a value for each year.
                                             Example UserInput: "What is the distribution of expenses by category? Give me the chart."
                                             Example Output: ["Food", "Transport", "Entertainment"]
                                             Give out the output, not the UserInput.
                                             Don't create new output, get it from {result_search}.
                                             Always take the label from the given data.
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

                                            You must not write anything like "Output: ", just write the title.
                                            UserInput: {question}.""")
    
    label_chain = prompt | llm | StrOutputParser()
    return label_chain

def get_data_chain():
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    prompt_template = """You are responsible for selecting the correct tool from the following list and specifying the function name and arguments based on the user's question.
                     Choose the correct tool from the list below, finding the one relevant to the question, and provide the function name and arguments to extract the relevant information from the context.
                     The number of elements in the lists must be equal to the number of labels.
                     Only output numeric values, corresponding to prices.
                     ALWAYS EXTRACT NUMBERS AT THE END.
                     Never extract string values.
                     through=for each
                     over=for each
                     most_expensive=get_price
                     Select the prices from the context.
                     Here are the tools available and their descriptions:
                     {rendered_tools}
                     Your goal:
                     - Only get numeric values.
                     - Never get values that are strings.
                     - You must not provide all 0.0 as data.
                     - Select the correct tool for the task.
                     - If no specific time frame is provided, please start from today and extend back to the earliest year you can imagine                     - Ensure **every** part of the context is passed to the tools.
                     - Provide a response in JSON format with 'name' and 'arguments' keys.
                     - If "average" or "mean" in context, use the average tool.
                     - You must get one value for each label.
                     - Don't get more than one value for each label.
                     - If you have a time indication, do not ignore it. Questions like "through the years" should have a value for each year.
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


def get_graph_type(type_chain, question):
    return type_chain.invoke({"question": question})

def get_labels(lables_chain, question, chart_type, result_search, PRINT_SETTINGS):
    labels = lables_chain.invoke({"question": question, "chart_type": chart_type, "result_search": result_search})
    
    # Check if the input is a list-like string
    if labels.startswith("[") and labels.endswith("]"):
        try:
            labels = ast.literal_eval(labels)
        except (ValueError, SyntaxError):
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

    data_dict = {}
    if isinstance(context[0], str) and 'T' in context[0]:  # Check for timestamp format (with 'T')
        # Data contains dates and values
        for i, entry in enumerate(context):
            if i % 2 == 0:  # This is a key
                key = entry[:10].strip('"')
                value = context[i + 1]
                if key in data_dict:
                    data_dict[key].append(value)  # Append value to existing list
                else:
                    data_dict[key] = [value]  # Create a new list for new key
    else:
        # Data contains simple labels and values
        for i in range(0, len(context), 2):
            key = context[i].strip('"')
            value = context[i + 1]
            if key in data_dict:
                data_dict[key].append(value)  # Append value to existing list
            else:
                data_dict[key] = [value]  # Create a new list for new key

    aligned_data = []
    
    print(f"Data_Dict: {data_dict}")
    # Iterate over each key-value pair in the dictionary
    labels = [label.strip('"').strip("'").lower() for label in labels]
    for key, values in data_dict.items():
        if key.lower() in labels:
            for value in values:
                aligned_data.append(value)
        else:
            for value in values:
                if value in labels:
                    aligned_data.append(key)

    if (PRINT_SETTINGS["print_plot_data"]):
        print(f"Aligned Data: {aligned_data}")
    return aligned_data



def get_label_title(question, model):
    return model.invoke({"question": question})

def get_chart_description_chain():
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)

    prompt_template = """You are an expert at financial analysis. You will receive a Chart.js chart, divided into its data and config parameters.
                        You will output a description of the chart, giving insights about it if you have any.
                        For example, if you receive a line chart you could say where you have minimums and maximums.
                        If you receive a pie chart, you could say which category is the most represented.
                        If you receive a bar chart, you could say which category is the most expensive.
                        Invent other insights if you want, but don't invent anything that is not related to the graph.
                        Don't say things like "The chart is designed with a blue color scheme and has a scale that begins at zero, providing a clear visual representation of the expenditure."

                        Data: {data}
                        Config: {config}
                        Output only the insight and description, no other information.
                        Explain it only one time. Be concise."""

    prompt = ChatPromptTemplate.from_template(
        prompt_template
    )

    prompt = prompt.partial(
        data="{data}",
        config="{config}"
    )

    description_chain = prompt | llm | StrOutputParser()

    return description_chain

def write_chart_html(chart_type, labels, data, label, chart_description_chain, filename="chart.html"):
    background_colors = [
        'rgba(3, 7, 30, 0.5)',
        'rgba(55, 6, 23, 0.5)',
        'rgba(106, 4, 15, 0.5)',
        'rgba(157, 2, 8, 0.5)',
        'rgba(208, 0, 0, 0.5)',
        'rgba(220, 47, 2, 0.5)',
        'rgba(232, 93, 4, 0.5)',
        'rgba(244, 140, 6, 0.5)',
        'rgba(250, 163, 7, 0.5)',
        'rgba(255, 186, 8, 0.5)',
    ] * (len(labels) // 10 + 1)

    border_colors = [
        'rgba(3, 7, 30, 1)',
        'rgba(55, 6, 23, 1)',
        'rgba(106, 4, 15, 1)',
        'rgba(157, 2, 8, 1)',
        'rgba(208, 0, 0, 1)',
        'rgba(220, 47, 2, 1)',
        'rgba(232, 93, 4, 1)',
        'rgba(244, 140, 6, 1)',
        'rgba(250, 163, 7, 1)',
        'rgba(255, 186, 8, 1)',
    ] * (len(labels) // 10 + 1)

    if (chart_type != "line"):
        data = {
            'labels': labels,
            'datasets': [{
                'data': data,
                'backgroundColor': background_colors[:len(labels)],
                'borderColor': border_colors[:len(labels)],
                'borderWidth': 1
            }]
        }
    else:
        data = {
            'labels': labels,
            'datasets': [{
                'data': data,
                'backgroundColor': 'rgba(75, 192, 192, 0.5)',
                'borderColor': 'rgba(75, 192, 192, 1)',
                'borderWidth': 1
            }]
        }

    config = {
        'type': chart_type,
        'data': data,
        'options': {
            'plugins': {
                'legend': {
                    'display': True if chart_type == "pie" else False
                }
            },
            'responsive': True,
            'maintainAspectRatio': False,
            'scales': {
                'y': {
                    'beginAtZero': True,
                    'title': {
                        'display': chart_type != "pie",
                        'text': label if chart_type != "pie" else ""
                    }
                }
            } if chart_type != "pie" else {}
        }
    }

    chart_description = chart_description_chain.invoke({"data": data, "config": json.dumps(config)})

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{label} Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    body, html {{
        height: 100%;
        font-family: 'Arial', sans-serif;
        background-color: #eaeef1; /* Light background for contrast */
        color: #333; /* Dark text for readability */
    }}
    body {{
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        height: 100vh;
        position: relative;
        padding: 20px;
    }}
    .title-container {{
        margin-bottom: 20px;
        text-align: center;
        width: 80%;
        max-width: 900px;
        padding: 20px;
        background-color: #fff; 
        border-radius: 12px; 
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); 
    }}
    .title {{
        margin: 0; /* Remove default margins */
    }}
    .chart-container {{
        width: 80%;
        max-width: 900px;
        height: 70vh;
        background: #fff;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        z-index: 10;
        position: relative;
        transition: transform 0.2s; /* Smooth scaling effect */
    }}
    .chart-container:hover {{
        transform: scale(1.01); /* Slightly enlarge on hover */
    }}
    .description-container {{
        width: 80%;
        max-width: 900px;
        padding: 20px;
        margin-top: 20px;
        background-color: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        text-align: center;
        line-height: 1.5; /* Improve readability */
    }}
    .description-container:hover {{
        transform: scale(1.01); /* Slightly enlarge on hover */
    }}
    canvas {{
        display: block;
        border-radius: 8px; /* Rounded corners on the chart */
    }}
    h1 {{
        margin-bottom: 20px;
        font-size: 2em; /* Larger title */
        color: #2c3e50; /* Darker text color for the title */
    }}
    p {{
        font-size: 1.1em; /* Slightly larger font for descriptions */
        color: #555; /* Softer text color for descriptions */
    }}
</style>
</head>
<body>
<div class="title-container">
    <h1 class="title">{label} chart</h1>
</div>
    <!-- Chart section -->
    <div class="chart-container">
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
    <!-- Explanation section -->
    <div class="description-container">
        <p id="chart-description">{chart_description}</p>
    </div>
</body>
</html>
"""

    with open(filename, 'w') as f:
        f.write(html_content)

