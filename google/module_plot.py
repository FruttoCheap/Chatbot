import re
import pandas as pd
from pandasai import SmartDataframe
from pandasai import Agent
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
    agent = Agent(dataframe, config={"llm": llm}, description="""You are a data analysis agent. 
                                            Your main goal is to help non-technical users to analyze data on their financial expenses. 
                                            The charts must have numbers to ease the reading.
                                            Give an SVG file and not a PNG file. 
                                            Make sure everything is visible.
                                            Make it very good looking.
                                            If you need to display categories, display everything.
                                            Don't print anything to terminal and don't open newly generated SVG files.
                                            Don't use floats for time.
                                            Only one plot per question.
                                            """)
    sdf = SmartDataframe(dataframe, config={"llm": llm})
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
    
    agent.chat(question + """ Create an appropriate very good looking plotly chart using only the relevant data from the database.
                            Give an SVG file and not a PNG file. 
                            Make sure everything is visible.
                            Organize the spaces in a way that everything is readable. You don't have limits.
                            If you need to display categories, display everything.
                            Don't use floats for time.
                            Make the chart very good looking.
                            Only one plot per question.""")

    if (PRINT_SETTINGS["print_explaination_plot"]):
        res = agent.explain()
        parser = StrOutputParser()
        system_template = """You will receive a description of the procedure used to extract a plot from the data.
                             You will transform this description as if you were talking to a non-technical user, who only cares about which data got used
                             and about the kind of chart created.
                             The only thing you will talk about is what data you chose and what kind of chart you created.
                             Don't talk about databases, SQL, SVG, titles, labels or any technical stuff.
                             Don't talk about what the title and the labels are.
                             Be very short and concise.
                             
                             The description is: {description}"""
        prompt_template = ChatPromptTemplate.from_messages(
                          [("system", system_template), ("user", "{description}")]
                          )

        chain = prompt_template | llm | parser 
        res = chain.invoke({"description": res})
        print(f"Explaination: {res}")