from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

class Classification(BaseModel):
    is_there_time: int = Field(description="0 if there is nothing related to time in the question, 1 if there is.")
    is_specific: int = Field(description="From 0 to 10, how specific is the question?")
    is_broad: int = Field(description="From 0 to 10, how broad is the question?")
    has_keyword: int = Field(description="From 0 to 10, are there keywords in the question? If there are not, put 0.")
    is_related_to_personal_finance: int = Field(description="From 0 to 10, how is the question related to the expenses of the user? It is to be considered related if it asks anything about prices or expenses or items purchased. Put 0 if the question is not related to prices, expenses, money or items.")
    NLP_or_RAG: int = Field(description="0 if the question is better suited for RAG queries, 1 if the question is better suited for NLP queries. You can have a value between 0 and 10.")
    is_there_number: int = Field(description="0 if there is no explicit number in the question, 1 if there is and explicit number in the question.")
    is_there_category: int = Field(description="0 if the word 'category' is not in the question, 1 if there is the word 'category' in the question.")
    is_it_question: int = Field(description="1 if the question asks for something in return, 0 if it is just a sentence.")
    is_asking_for_plot: int = Field(description="1 if the question is asking in any way for a plot or a graph, 0 if it is not.")

def get_tagging_chain():
    tagging_prompt = ChatPromptTemplate.from_template("""
            Extract the desired information from the following passage.
            You will classify the passage based on the following criteria:
            You must put a value in each field.
                                                  
            1. **is_there_time**: Set to 1 if the passage mentions any time-related information (e.g., days, weeks, dates, months, time ranges), otherwise set to 0.
            2. **is_specific**: Rate the question's specificity on a scale from 0 to 10, where 0 means the question is very vague, and 10 means it is highly specific.
            3. **is_broad**: Rate the question's breadth on a scale from 0 to 10, where 0 means it is very narrow in scope, and 10 means it is very broad.
            4. **NLP_or_RAG**: Rate from 0 to 10, where 0 means the question is better suited for a RAG query (unstructured text), and 10 means it is better suited for NLP queries (structured data).
            5. **has_keyword**: Rate from 0 to 10, where 0 means there are no keywords in the question, and 10 means there are many keywords.
            6. **is_related_to_personal_finance**: Rate from 0 to 10, where 0 means the question is not related to personal finance, and 10 means it is highly related to personal finance. 
                                                   It is to be considered related if it asks anything about prices or expenses or items purchased.
                                                   Put 0 if the question is not related to prices, expenses, money or items.
            7. **is_there_number**: Set to 1 only if there an explicit number in the question, otherwise set to 0.
            8. **is_there_category**: Set to 1 only if there is the word 'category' in the question, otherwise set to 0.
            9. **is_it_question**: Set to 1 if the question asks for something in return, otherwise set to 0.
            10. **is_asking_for_plot**: Set to 1 if the question is asking in any way for a plot or a graph, otherwise set to 0.
            Passage:
            {input}
            """)

    llm = ChatGroq(temperature=0, model="llama3-groq-70b-8192-tool-use-preview").with_structured_output(Classification)

    tagging_chain = tagging_prompt | llm

    return tagging_chain

def get_classification(question, tagging_chain, PRINT_SETTINGS):
    try:
        class_calculed = tagging_chain.invoke({"input": question})
    except Exception:
        return "rejected (exception)"
    
    res_dict = class_calculed.dict()

    if PRINT_SETTINGS["print_characteristics_of_the_question"]:
        print(res_dict)

    if res_dict["is_related_to_personal_finance"] <= 2:
        return "REJECTED"
    elif res_dict["is_it_question"] == 0:
        return "INPUT"
    elif res_dict["is_there_time"] == 1 or res_dict["is_there_number"] == 1 or res_dict["is_there_category"] == 1:
        if res_dict["is_asking_for_plot"] == 1:
            return "NLP|PLT"
        else:
            return "NLP"
    elif (res_dict["is_specific"] == res_dict["is_broad"]) or res_dict["is_broad"] > 5 or res_dict["NLP_or_RAG"] > 5:
        if res_dict["is_asking_for_plot"] == 1:
            return "RAG|PLT"
        return "RAG"
    else:
        return "NLP"
