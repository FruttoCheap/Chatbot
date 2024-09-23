# tried to get numbers to explicit characteristics of the question. Don't really like the results

import os
from time import sleep
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

class Classification(BaseModel):
    is_there_time: int = Field(description="0 if there is nothing related to time in the questino, 1 if there is.")
    is_specific: int = Field(description="From 0 to 10, how specific is the question?")
    is_broad: int = Field(description="From 0 to 10, how broad is the question?")
    has_keyword: int = Field(description="From 0 to 10, are there keywords in the question? If there are not, put 0.")
    is_related_to_personal_finance: int = Field(description="From 0 to 10, how is the question related to the expenses of the user? If it is not, put 0.")
    NLP_or_RAG: int = Field(description="0 if the question is better suited for RAG queries, 10 if the question is better suited for NLP queries. You can have a value between 0 and 10.")

# Environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

tagging_prompt = ChatPromptTemplate.from_template("""
            Extract the desired information from the following passage.
            You will classify the passage based on the following criteria:

            1. **is_there_time**: Set to 1 if the passage mentions any time-related information (e.g., dates, months, time ranges), otherwise set to 0.
            2. **is_specific**: Rate the question's specificity on a scale from 0 to 10, where 0 means the question is very vague, and 10 means it is highly specific.
            3. **is_broad**: Rate the question's breadth on a scale from 0 to 10, where 0 means it is very narrow in scope, and 10 means it is very broad.
            4. **NLP_or_RAG**: Rate from 0 to 10, where 0 means the question is better suited for a RAG query (unstructured text), and 10 means it is better suited for NLP queries (structured data).
            5. **has_keyword**: Rate from 0 to 10, where 0 means there are no keywords in the question, and 10 means there are many keywords.
            6. **is_related_to_personal_finance**: Rate from 0 to 10, where 0 means the question is not related to personal finance, and 10 means it is highly related to personal finance.
                                                              
            Passage:
            {input}
            """)

# LLM
llm = ChatGroq(temperature=0, model="llama3-groq-70b-8192-tool-use-preview").with_structured_output(Classification)

tagging_chain = tagging_prompt | llm

def get_classification(q):
    try:
        class_calculed = tagging_chain.invoke({"input": q})
    except Exception as e:
        print(e)
        return "I'm sorry, I couldn't classify the question."
    
    print(class_calculed.dict())


listOfQuestions1 = [
    "How much did I spend on yoga?",                                                # ok
    "Find all purchases related to 'fashion' made in September 2023.",              # ok
    "How much did I spend on clothing?",                                            # ok   
    "How much did I spend in 2023?",                                                # ok
    "How much did I spend in 2020?",                                                # ok
    "How much did I spend while hanging out?",                                      # ok, difficult to say
    "On average, how much do I spend daily?",                                       # ok
    "What was the most expensive item in the 'electronics' category?",              # ok
    "How much did the leather jacket cost?",                                        # ok
    "What is the total spent on items in the 'food' category?",                     # ok
    "List all items purchased in the 'entertainment' category.",                    # ok
    "How many electronics items were purchased after September 15th, 2023?",        # ok
    "Which purchase was made on 2023-09-21?",                                       # ok   
    "What is the price of the 'designer handbag'?",                                 # ok 
    "How many purchases were made in the 'health' category in October?",            # ok
    "What was the price of the 'fancy dinner'?",                                    # ok
    "What was bought on 2023-10-11?",                                               # ok                   
    "How many items were purchased in the 'travel' category?",                      # ok
    "Which item was the cheapest?"                                                  # ok
]

listOfQuestions2 = [
    "How many items were bought in the afternoon (12 PM - 6 PM)?",                  # not ok
    "How much was spent on 'furniture' items in total?",                            # ok
    "What is the average price of all 'electronics' items?",                        # ok
    "What item was purchased on 2023-10-01?",                                       # ok
    "What are all the items bought in the 'kitchen' category?",                     # ok
    "How many purchases were made in the 'automotive' category?",                   # ok
    "What item was bought on 2023-10-13?",                                          # ok
    "Find all items purchased on weekends (Saturday and Sunday).",                  # ok
    "What is the price of the 'new laptop'?",                                       # ok
    "What is the earliest purchase in the dataset?",                                # ok
    "Which category had the most items purchased in October?",                      # ok
    "What was the most expensive purchase in the 'health' category?",               # ok
    "How much was spent on all purchases made in October 2023?",                    # ok
    "What was purchased on 2023-09-05?",                                            # ok
    "How many items were purchased in the 'decor' category?",                       # ok
    "What was the total spent on the 'weekend trip' and 'suitcase' combined?",      # ok
    "What was the least expensive 'electronics' item?",                             # ok
    "How many purchases were made after 5 PM?",                                     # ok
    "Which item purchased in September cost exactly 100 units?",                    # ok
    "How many 'food' items were bought after September 15th, 2023?",                # ok
    "What is the price of the 'digital camera'?",                                   # ok
    "How many purchases were made in the 'furniture' category?",                    # ok
    "What was bought for 85 units?",                                                # ok
    "How much was spent on the 'tablet' and the 'gaming console' combined?",        # ok
    "What was the most expensive item purchased in the 'travel' category?",         # ok
    "Find all purchases made in the morning (before 12 PM).",                       # ok
    "How many items were bought for less than 100 units?",                          # not ok, but easy for NLP
    "What is the total amount spent on all items?",                                 # not ok, but easy for NLP
    "Which item in October was purchased for exactly 550 units?",                   # ok
    "What was the price of the 'new set of tires'?",                                # ok
    "How many items were bought for more than 300 units?",                          # not ok, but easy for NLP
    "What is the total number of purchases made in October 2023?",                  # ok
    "What item was bought for 110 units?",                                          # ok
    "How much did the 'brand new watch' cost?",                                     # ok
    "What is the price of the 'pair of wireless earbuds'?",                         # ok
    "What was the price of the 'book' bought in October 2023?"                      # ok
]

for i in listOfQuestions1 + listOfQuestions2:
    print(i)
    get_classification(i)
    print("---")