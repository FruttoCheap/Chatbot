import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from time import sleep
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Function used to elaborate queries
def get_from_database(printQuestion, printQuery, needToWait, question, db, chain, today, time, cur_year):
    
    if "item" in question:
        question = question.replace("item", "thing")
        if "expensive" in question:
            question = question + " Need the description AND the price. Use ORDER BY"
    if "items" in question:
        question = question.replace("items", "things")
    if "category" in question:
        question = question + " Select only one category."
    if "categories" in question:
        question = question + " Select all distinct categories without limits."
    if "percentage" in question:
        question = question + " Use divisions/SUM and moltiplication*100."
    response = chain.invoke({"question": question})

    response = response.replace("SQLQuery:", "").replace("```sql", "").replace("```", "").replace("\n",";").strip()

    if (printQuestion):
        print(question)

    if (printQuery):
        print(response)

    result = db.run(response[response.find("SELECT"):])
    result = result.replace("[(","").replace(",)]","").replace("(","").replace(")","").replace(",,",",").replace("'","").replace("]","")

    if not result:
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

    if (needToWait):
        sleep(1)
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# define the chain

db = SQLDatabase.from_uri("sqlite:///googleDb.sqlite3")
llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_retries=2,
)
db.run("DROP TABLE IF EXISTS 'EXPENSES';")
chain = create_sql_query_chain(llm, db)
today = datetime.today().strftime('%Y-%m-%d')
time = datetime.today().strftime('%H:%M:%S')
cur_year = datetime.now().year
system = """
            You are given a database where there are all the items/purchases done by a person. There are also corresponding dates.
            - Use SQLITE3 syntax.
            Follow this rules:
            - only query existing tables
            - Use current date only if other time is not given.
            - Month Year: >= <=
            - The present year is {cur_year}
            - If you need to subtract time from a date, use a syntax like date('2024-09-12', '-30 days').
            - "What is the least expensive item bought" and similiar requests want the description (yes), not the price (no) 
            - Don't use the INTERVAL keyword: adjust the date calculations using strftime or date functions supported by SQLite.
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
            - Questions containing 'from month to month' should have two BETWEEN.
            - Output the queries only.
            """
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{query}")]
).partial(dialect=db.dialect,cur_year=cur_year, time=time, today=today)
validation_chain = prompt | llm | StrOutputParser()
full_chain = {"query": chain} | validation_chain

# Test Questions
questions1ok = [                           
    "How many items I bought in September 2023?",                                                                       # ok hardcoded
    "Which category has the highest number of purchases on weekends, and what is the total amount spent?",              # ok hardcoded
    "Which categories have purchases made on weekends?",                                                                # ok hardcoded 
    "What percentage of total spending is attributed to electronics items?",                                            # ok hardcoded
    "Which category has the highest average price per item?",                                                           # ok hardcoded
    "What is the least expensive item bought in the food category?",                                                    # ok hardcoded
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
    "What is the average price difference between fashion and electronics categories?",                                 # ok
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
for j in questions1ok:
    get_from_database(True, True, True, j, db, full_chain, today, time, cur_year)
