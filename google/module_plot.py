import re
import pandas as pd
from pandasai import SmartDataframe
from pandasai import Agent
from langchain_groq import ChatGroq

def get_plot_model():
    return ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)

# It should receive the queries only, not the actual data
def get_plot_from_all(llm_plot, db, question, PRINT_SETTINGS):
    all_data = pd.read_sql_query("SELECT * from expensesok;", db)
    get_plot(all_data, question, llm_plot, PRINT_SETTINGS, is_there_time=True)

def get_plot_from_RAG(llm_plot, search_output, question, PRINT_SETTINGS):
    lines = search_output.split('\n')
    pattern = r'(\d+\.\d+)\s+(.+)\s+(\w+)$'
    data = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            price = float(match.group(1))
            description = match.group(2)
            category = match.group(3)
            data.append({
                'price': price,
                'description': description,
                'category': category
            })
    df = pd.DataFrame(data)
    
    get_plot(df, question, llm_plot, PRINT_SETTINGS, is_there_time=False)

def get_plot(dataframe, question, llm, PRINT_SETTINGS, is_there_time):
    agent = Agent(dataframe, description="""You are a data analysis agent. 
                                            Your main goal is to help non-technical users to analyze data on their financial expenses. 
                                            The charts must have numbers to ease the reading. 
                                            The explaination must be very short and regard only the data retrieved and the chart used.""")
    sdf = SmartDataframe(dataframe)
    if (is_there_time):
        sdf = SmartDataframe(sdf, name="Financial Expenses", 
                                  description="""The database contains the columns price, description, category, timestamp. 
                                                 Each row is an expense of the user. 
                                                 You will be called to plot the correct data, given the question.""",
                                  config={"llm": llm})
    else:
        sdf = SmartDataframe(sdf, name="Financial Expenses", 
                                  description="""The database contains the columns price, description, category. 
                                                 Each row is an expense of the user. 
                                                 You will be called to plot the correct data, given the question.""",
                                  config={"llm": llm})
    agent.chat(question + " Create an appropriate plotly chart using only the relevant data from the database.")

    if (PRINT_SETTINGS["print_explaination_plot"]):
        print(agent.explain())