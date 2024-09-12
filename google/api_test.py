import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_groq import ChatGroq
from pydantic.v1 import BaseModel, Field
import firebase_admin
from firebase_admin import firestore

load_dotenv()

appFirebase = firebase_admin.initialize_app()
db = firestore.client()

app = FastAPI()


class Answer(BaseModel):
    """Outputs the structured version of the user input."""

    price: float = Field(description="The sum of money spent by the user")
    description: str = Field(description="A short description of the user's expense")
    category: str = Field(description="The category of the user's expense")


# model = OllamaFunctions(
#     model="google",
#     keep_alive=-1,
#     format="json",
# )

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_retries=2,
)


def get_prompts(q):
    system_prompt = """You are a finance manager who will receive a textual input after the tag UserInput. You must 
    organize the UserInput into the following schema: float price, string description, string category. The price and 
    the description must be taken directly from the input, while the category must be chosen among: Clothing, Food, 
    Health, Entertainment, Transport, Bills, Other."""

    template = f"""UserInput: {q}."""
    return [("system", system_prompt), ("human", template)]


def get_chain():
    return model.with_structured_output(Answer)


def get_answer(q):
    llm = get_chain()
    result = llm.invoke(get_prompts(q))
    return result


@app.post("/text/")
def create_item(text: str, uid: str):
    answer = get_answer(text)
    time = datetime.now()
    try:
        (db.collection("users")
         .document(uid)
         .collection("expenses")
         .add({"price": answer.price, "description": answer.description, "category": answer.category, "time": time}))
        return {""}
    except Exception as e:
        return {"error": e}
