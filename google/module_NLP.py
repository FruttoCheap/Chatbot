from timeit import default_timer as timer
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Obtain the database
def get_database(uri):
    db = SQLDatabase.from_uri(uri)
    db.run("DROP TABLE IF EXISTS 'EXPENSES';")
    return db

# Function used to elaborate queries-
def NLP(question, db, chain, correction_chain, description_chain, PRINT_SETTINGS):
    start = timer()
    if PRINT_SETTINGS["print_question"]:
        print(f"Question: {question}")

    response = chain.invoke({"question": question})
    response = response.replace("SQLQuery:", "").replace("```sql", "").replace("```", "").replace("\n",";").strip()

    if PRINT_SETTINGS["print_query"]:
        print(f"Original Query: {response}")

    today = datetime.today().strftime('%Y-%m-%d')
    response = correction_chain.invoke({"query":response, "question":question, "today":today}).strip()

    if PRINT_SETTINGS["print_corrected_query"]:
        print(f"Corrected query: {response}")

    result = db.run(response[response.find("SELECT"):])
    result = result.replace("[(","").replace(",)]","").replace("(","").replace(")","").replace(",,",",").replace("'","").replace("]","")

    if PRINT_SETTINGS["print_description"]:
        queryDescription = description_chain.invoke({"query":response})
        print(queryDescription)

    if not result or result=="None":
        return "No match found"
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
    
    end = timer()

    if PRINT_SETTINGS["print_time"]:
        print(f"Time: {end-start}")

    return result

# Function used to elaborate queries
def get_NLP_chains(db):
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    chain = create_sql_query_chain(llm, db)
    today = datetime.today().strftime('%Y-%m-%d')
    time = datetime.today().strftime('%H:%M:%S')
    cur_year = datetime.now().year
    system = """
                You are given a database where there are all the items/purchases done by a person. 
                You only have the columns price, description, category, and timestamp (which contains both day and time in isoformat).
                Whatever you do, you must not output the word INTERVAL.
                Whatever you do, you must not use syntax like date('2024-09-13') - 7 days.
                Whatever you do, you must not subtract from dates.
                If the question regards anything about plots, you must only select 2 columns.
                - Use SQLITE3 syntax.
                Follow these rules:
                - TOP: ORDER BY DESC
                - Never use LIMIT 5 if not actually required.
                - Use LIMIT 1 only if you must find the min or max value.
                - For questions like "How much do I spend in the evenings?" you should output the total spending after 18:00 from the first day. 
                - If the question is asked in present tense, start from the first day to today.
                - Use current date only if no other time is given.
                - For questions like "How much did I spend on yoga?" the description should contain 'yoga', not be entirely it.
                - The present year is {cur_year}, today is {today}
                - "What is the least expensive item bought" and similar requests want the description (yes), not the price (no) 
                - Don't use BETWEEN: go for direct comparisons.
                - If you need to go in the past, remember that today is {today}.
                - How many: COUNT
                - Questions SIMILAR TO 'What is the least expensive item' MUST output an item.
                - If there are singular names, the probability of a single row is higher. 
                - MUST Suppose the year {cur_year}, UNLESS not already given in the question. 
                - If no year/month/day/time is provided, test all possible year/month/day/time. 
                - Don't use built-ins unless comparing dates: today is {today}, time is {time}. 
                - Use time only when specified (e.g., 6 PM, 18:00). 
                - For date comparisons, prefer >= or <=. 
                - If time is given without a date, use only the time. 
                - Use DISTINCT for category queries.
                - Output the queries only.
                """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{query}")]
    ).partial(dialect=db.dialect,cur_year=cur_year, time=time, today=today)
    validation_chain = prompt | llm | StrOutputParser()
    full_chain = {"query": chain} | validation_chain

    # define the chain that will correct

    llm2 = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    parser = StrOutputParser()
    template = ChatPromptTemplate.from_messages([("system", """You are a SQLite3 query checker. You will receive an SQLite3 query here {query} and correct it syntattically.
                                                The query should respond to this question: {question} Respond only with the corrected query. 
                                                Always remove INTERVAL keyword: adjust the date calculations using strftime or date functions supported by SQLite. 
                                                Never use INTERVAL keyword: adjust the date calculations using strftime or date functions supported by SQLite.
                                                Never use INTERVAL keyword: adjust the date calculations using strftime or date functions supported by SQLite.
                                                Never use INTERVAL keyword: adjust the date calculations using strftime or date functions supported by SQLite.
                                                If the question regards plots, the query must only select 2 columns.
                                                Correct syntax: date('2024-09-12', '-30 days')
                                                Correct syntax: date('2024-09-12', '+1 day')
                                                Never use LIMIT 5 if not actually required.
                                                - TOP: ORDER BY DESC
                                                Lowest: group by ORDER BY ASC
                                                Least: group by ORDER BY ASC
                                                If there are words ending in "-est", LIMIT 1.
                                                The table name is expensesok. 
                                                "What's the category I spent the lowest?" SELECT "category", SUM("price") AS "total_spent" FROM expensesok GROUP BY "category" ORDER BY "total_spent" DESC LIMIT 1;
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

    llm3 = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    parser2 = StrOutputParser()
    template2 = ChatPromptTemplate.from_messages([("system", """You will receive an SQL3 query and a result. You will describe what the query gets to a user that does not know anything about databases, as if the database and the query did not exist. I only see the result. The query is {query}. Give a one line result. Don't talk about the result and the query. Template: The search found: (short description of what that query should find).""")])  
    description_chain = template2 | llm3 | parser2

    return full_chain, correction_chain, description_chain
