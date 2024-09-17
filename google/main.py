import os
from timeit import default_timer as timer
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Function used to elaborate queries
def get_from_database(printQuestion, printQuery, printDescription, printCorrectedQuery, question, db, chain, correction_chain, today, description_chain):
    start = timer()
    if (printQuestion):
        print(f"Question: {question}")

    response = chain.invoke({"question": question})
    response = response.replace("SQLQuery:", "").replace("```sql", "").replace("```", "").replace("\n",";").strip()

    if (printQuery):
        print(f"Original Query: {response}")

    response = correction_chain.invoke({"query":response, "question":question, "today":today}).strip()

    if (printCorrectedQuery):
        print(f"Corrected query: {response}")

    result = db.run(response[response.find("SELECT"):])
    result = result.replace("[(","").replace(",)]","").replace("(","").replace(")","").replace(",,",",").replace("'","").replace("]","")

    if (printDescription):
        queryDescritption = description_chain.invoke({"query":response})
        print(queryDescritption)

    if not result or result=="None":
        print("No match found.")
    else:
        # rounding decimals to the second place
        for word in result.split():
            if "." in word:
                try:
                    fl = float(word)
                    fl = round(fl, 2)
                    fl = format(fl, '.2f')
                    result = result.replace(word, str(fl))
                except ValueError:
                    continue
        print(result)
    
    
    print("---")

    end = timer()

    return (end-start)

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# define the chain that will correct

llm2 = Ollama(model="gemma2:2b")
parser = StrOutputParser()
template = ChatPromptTemplate.from_messages([("system", """You are a SQLite3 query checker. You will receive an SQLite3 query here {query} and correct it syntattically.
                                              The query should respond to this question: {question} Respond only with the corrected query. 
                                              Always remove INTERVAL keyword: adjust the date calculations using strftime or date functions supported by SQLite. 
                                              Never use INTERVAL keyword: adjust the date calculations using strftime or date functions supported by SQLite.
                                              Never use INTERVAL keyword: adjust the date calculations using strftime or date functions supported by SQLite.
                                              Never use INTERVAL keyword: adjust the date calculations using strftime or date functions supported by SQLite.
                                              Correct syntax: date('2024-09-12', '-30 days')
                                              Correct syntax: date('2024-09-12', '+1 day')
                                              Correct syntax: date('2024-09-12', '-30 days')
                                              Correct syntax: date('2024-09-12', '+1 day')
                                              Correct syntax: date('2024-09-12', '-30 days')
                                              Correct syntax: date('2024-09-12', '+1 day')
                                              Correct syntax: date('2024-09-12', '-30 days')
                                              Correct syntax: date('2024-09-12', '+1 day')
                                              The table name is expensesok. 
                                              How much: SUM()
                                              How much: SUM()
                                              How much: SUM()
                                              In which: ORDER BY
                                              present tense: NO date()
                                              present tense: NO date()
                                              You only have columns: price, description, category, timestamp (which include date and time)
                                              If the date of today is needed, remember that today is {today}. 
                                              NEVER subract numbers to dates. """)])  
correction_chain = template | llm2 | parser

# define the description chain

llm3 = Ollama(model="gemma2:2b")
parser2 = StrOutputParser()
template2 = ChatPromptTemplate.from_messages([("system", """You will receive an SQL3 query and a result. You will describe what the query gets to me, as if the database and the query did not exist. I only see the result. The query is {query}. Give a one line result. Don't talk about the result and the query. Template: The search found: (short description of what that query should find).""")])  
description_chain = template2 | llm3 | parser2


# define the chain that will generate

db = SQLDatabase.from_uri("sqlite:///googleDb.sqlite3")
llm = Ollama(model="gemma2:2b")
db.run("DROP TABLE IF EXISTS 'EXPENSES';")
chain = create_sql_query_chain(llm, db)
today = datetime.today().strftime('%Y-%m-%d')
time = datetime.today().strftime('%H:%M:%S')
cur_year = datetime.now().year
system = """
            You are given a database where there are all the items/purchases done by a person. 
            You only have the columns price, description, category and timestamp (which contains both day and time in isoformat)
            Whatever you do, you must not output the word INTERVAL.
            Whatever you do, you must not use syntax like date('2024-09-13') - 7 days.
            Whatever you do, you must not subract to dates.
            - Use SQLITE3 syntax.
            Follow this rules:
            - only query existing tables
            - For questions like "How much do I spend in the evenings?" you should output the total spending after 18:00 from the first day. 
            - If the question is asked in present tense, start from the first day to today.
            - Use current date only if other time is not given.
            - For questions like "How much did I spend on yoga?" you the description should contain 'yoga', not be entirely it.
            - The present year is {cur_year}
            - "What is the least expensive item bought" and similiar requests want the description (yes), not the price (no) 
            - Don't use BETWEEN: go for direct comparisons.
            - If you need to go in the past, remember that today is {today}.
            - How many: COUNT 
            - Questions SIMILIAR TO 'What is the least expensive item' MUST output an item.
            - If there are singular names, the probability of a single row is higher. 
            - MUST Suppose the year {cur_year}, UNLESS not already given in the question. 
            - If no year/month/day/time is provided, test all possible year/month/day/time. 
            - Don't use built-ins unless comparing dates: today is {today}, time is {time}. 
            - Use time only when specified (e.g., 6 PM, 18:00). 
            - For date comparisons, prefer >= or <=. 
            - If time is given without a date, use only the time. 
            - Use DISTINCT for category queries. 
            - evenings or mornings without date do not care about the date but only about the
            - Output the queries only.
            """
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{query}")]
).partial(dialect=db.dialect,cur_year=cur_year, time=time, today=today)
validation_chain = prompt | llm | StrOutputParser()
full_chain = {"query": chain} | validation_chain

# Test Questions
questions1ok = [                           
    "How many items I bought in September 2023?",                                                                       # ok
    "Which category has the highest number of purchases on weekends, and what is the total amount spent?",              # ok
    "Which categories have purchases made on weekends?",                                                                # ok 
    "What percentage of total spending is attributed to electronics items?",                                            # ok
    "Which category has the highest average price per item?",                                                           # ok
    "What is the least expensive item bought in the food category?",                                                    # ok
    "Which category had the highest increase in spending from September to October?",                                   # not ok
    "What is the most expensive item I bought in the last month, and when did I buy it?",                               # ok
    "What is the total amount spent on entertainment?",                                                                 # ok
    "What is the average price of items in the electronics category?",                                                  # ok 
    "What is the total amount spent on fashion items?",                                                                 # ok
    "How many purchases were made in the health category?",                                                             # ok
    "What is the average price of all items purchased?",                                                                # ok
    "What was the most expensive item bought in October 2023?",                                                         # ok
    "How many items cost more than $100?",                                                                              # ok
    "What is the total amount spent on travel-related purchases?",                                                      # ok
    "How many items were bought in the electronics category?",                                                          # ok
    "Which item has the earliest purchase date?",                                                                       # ok
    "How many purchases cost less than $50?",                                                                           # ok
    "What was the most expensive purchase made?",                                                                       # ok
    "Which category has the highest total spending?",                                                                   # ok
    "What is the total amount spent on furniture items?",                                                               # ok
    "Which category has the most purchases?",                                                                           # ok
    "What is the total amount spent on fashion items rounded to the nearest cent?",                                     # ok
    "What is the most expensive fashion item bought?",                                                                  # ok
    "Which item has the earliest purchase date, and what category does it belong to?",                                  # ok
    "What is the total number of purchases made between $50 and $100?",                                                 # ok
    "Which three categories have the highest total spending combined?",                                                 # ok
    "What is the distribution of purchases by month and category?",                                                     # ok
    "What is the average price difference between fashion and electronics categories?",                                 # not ok
    "What is the average spending per transaction made on a Saturday?",                                                 # ok 
    "Which items were purchased in the last week, and what are their respective categories and prices?",                # ok
    "What is the total amount spent on items purchased after 6 PM?",                                                    # ok
    "What are my categories?",                                                                                          # ok
    "What is the total amount spent on purchases made before 10 AM?",                                                   # ok
    "What is the range of prices for items in the health category?",                                                    # ok
    "How many purchases were made after October 1, 2023?",                                                              # ok
    "Give me all purchases in 2023.",                                                                                   # ok
    "What is the most frequent purchase day of the week, and how much was spent on that day in total?",                 # ok but weird format
]

newQuestions = [
    "How much did I spent last month?",                                                                                 # ok
    "What was my most expensive purchase of all time?",                                                                 # ok
    "How much did I spent last year?",                                                                                  # ok
    "How much did I spent today?",                                                                                      # ok
    "How much money do I spend on the weekends?",                                                                       # not ok
    "How much money do I spend during non weekend days?",                                                               # ok
    "In which category I spend the biggest amount of money?",                                                           # ok
    "How much do I spend in the mornings?",                                                                             # ok
    "How much do I spend in the evenings?",                                                                             # ok
    "How much did I spend on yoga?",                                                                                    # ok
]

max_time = 0
total_time = 0
for j in questions1ok + newQuestions:
    x = get_from_database(True, True, True, True, j, db, full_chain, correction_chain, today, description_chain)
    total_time += x
    if x > max_time:
        max_time = x

print(f"The maximum time for a result was: {x:.2f} seconds.")
print(f"The average waiting time was: {(total_time/len(questions1ok+newQuestions)):.2f} seconds.")