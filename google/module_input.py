from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from datetime import datetime
from pydantic import ValidationError

class Answer(BaseModel):
    """Outputs the structured version of the user input."""

    price: float = Field(description="The sum of money spent by the user")
    description: str = Field(description="A short description of the user's expense")
    category: str = Field(description="The category of the user's expense")


def get_input_chain():
    llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0)
    prompt = PromptTemplate.from_template("You are a finance manager who will receive, from the user, a textual input. You must organize the UserInput into the following schema: float price, string description, string category. The price and the description must be taken directly from the input: remember to remove the articles. The category must be chosen among: Clothing, Food, Health, Personal Care, Entertainment, Transport, Bills, Subscription, Other. UserInput: {question}.")
    structured_llm = llm.with_structured_output(Answer)
    chain = prompt | structured_llm

    return chain


def input_into_database(question, chain, cur, MAX_DESCRIPTION_LENGTH):
    try:
        date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        # Generate structured response
        resp = chain.invoke({"question": question})
        
        # Input validation
        if resp.price <= 0:
            raise ValueError("Price must be provided as a positive value.")
        if not resp.description.strip():
            raise ValueError("A description of the expense must be provided.")
        valid_categories = {"Clothing", "Food", "Health", "Personal Care", "Entertainment", "Transport", "Bills", "Subscription", "Other"}
        if resp.category not in valid_categories:
            raise ValueError("Invalid category.")
        
        # Insert the new record into the database
        cur.execute("INSERT INTO expensesok (price, description, category, timestamp) VALUES (?, ?, ?, ?);", 
                    (resp.price, resp.description, resp.category, date))
        
        # Commit the transaction to save the changes
        cur.connection.commit()
        
        # Fetch the newly inserted record to confirm
        cur.execute("SELECT * FROM expensesok WHERE timestamp = ?;", (date,))
        result = cur.fetchone()
        
        return f"The following input has been added to the database:\nPrice: {result[0]}\nDescription: {result[1]}\nCategory: {result[2]}\nTimestamp: {result[3]}"
    
    except ValidationError as ve:
        return f"Input validation error: {ve}"
    except ValueError as e:
        return f"Value error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"