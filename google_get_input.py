import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic.v1 import BaseModel, Field

con = sqlite3.connect("googleDb.sqlite3")
cur = con.cursor()

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class Answer(BaseModel):
    """Outputs the structured version of the user input."""

    price: float = Field(description="The sum of money spent by the user")
    description: str = Field(description="A short description of the user's expense")
    category: str = Field(description="The category of the user's expense")


llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=GOOGLE_API_KEY, temperature=0.01)

prompt = PromptTemplate.from_template("You are a finance manager who will receive, from the user, a textual input. You must organize the UserInput into the following schema: float price, string description, string category. The price and the description must be taken directly from the input, while the category must be chosen among: Clothing, Food, Health, Entertainment, Transport, Bills, Other. UserInput: {question}.")

structured_llm = llm.with_structured_output(Answer)

chain = prompt | structured_llm

if __name__ == "__main__":
    sentences = [
        "I spent 100 dollars on a new pair of shoes.",
        "I just dropped 150 bucks on a leather jacket.",
        "I paid $85 for a gym membership.",
        "I shelled out $120 on a brand new watch.",
        "I spent $200 on a fancy dinner.",
        "I used $50 to buy a set of headphones.",
        "I purchased a coffee machine for $110.",
        "I spent 75 dollars on a new bookshelf.",
        "I paid $45 for a concert ticket.",
        "I dropped $250 on a designer handbag.",
        "I bought a new laptop for $900.",
        "I spent $30 on a bottle of wine.",
        "I paid 60 dollars for a pair of sneakers.",
        "I purchased a gaming console for $300.",
        "I shelled out $130 on a digital camera.",
        "I bought an art print for $40.",
        "I just spent $500 on a weekend trip.",
        "I used $70 to buy a skincare set.",
        "I paid $220 for a new desk chair.",
        "I spent $15 on a book at the store.",
        "I dropped 300 dollars on a tablet.",
        "I used $25 to buy a kitchen knife.",
        "I spent $80 on a massage session.",
        "I paid $40 for a plant for my apartment.",
        "I bought a new set of tires for $400.",
        "I just spent $60 on a board game.",
        "I purchased a suitcase for $180.",
        "I dropped $95 on a nice dinner with friends.",
        "I shelled out $550 for a phone upgrade.",
        "I spent $65 on a pair of wireless earbuds.",
        "I bought a ticket to a Broadway show for $125.",
        "I spent $10 on a cup of coffee and a pastry.",
        "I paid $40 for a yoga class.",
        "I used $100 to buy a bicycle.",
        "I dropped $300 on a smartwatch.",
        "I purchased a winter coat for $220.",
        "I just spent $25 on a movie ticket and popcorn.",
        "I shelled out $75 for a pair of jeans.",
        "I bought a blender for $90.",
        "I spent $135 on a weekend spa package.",
        "I paid $600 for a home theater system.",
        "I used $50 to buy a nice bottle of whiskey.",
        "I dropped $80 on a haircut and style.",
        "I spent $35 on a houseplant.",
        "I paid $95 for a pair of running shoes.",
        "I just bought a painting for $150.",
        "I shelled out $400 on a flat-screen TV.",
        "I spent $90 on a gym outfit.",
        "I used $15 to buy a new phone case.",
        "I dropped $200 on a weekend getaway.",
        "I purchased a cookbook for $30.",
        "I spent $500 on a DSLR camera.",
        "I paid $18 for a bottle of premium olive oil.",
        "I used $65 to buy a pair of wireless speakers.",
        "I bought a hiking backpack for $120.",
        "I dropped $45 on a new video game.",
        "I spent $250 on a coffee table.",
        "I paid $75 for a meal at a nice restaurant.",
        "I purchased a new pair of glasses for $180.",
        "I spent $300 on a flight ticket.",
        "I shelled out $200 on a smartphone.",
        "I dropped $28 on a sushi dinner.",
        "I used $95 to buy a smartwatch band.",
        "I just spent $60 on a new jacket.",
        "I paid $20 for a monthly subscription box.",
        "I bought a camera lens for $350.",
        "I spent $12 on a pack of craft beer.",
        "I used $400 to buy a used car.",
        "I dropped $75 on a pair of sandals.",
        "I paid $22 for a pair of gloves.",
        "I spent $85 on a bottle of cologne.",
        "I purchased a necklace for $150.",
        "I just spent $500 on a vacation package.",
        "I shelled out $40 on a Bluetooth speaker.",
        "I used $35 to buy a new pillow.",
        "I bought a yoga mat for $25.",
        "I spent $70 on a fancy dinner date.",
        "I paid $45 for a new shirt.",
        "I dropped $300 on a bicycle repair.",
        "I purchased a leather wallet for $90.",
        "I spent $400 on a video game console.",
        "I just spent $30 on a wireless mouse.",
        "I used $55 to buy a dinner for two.",
        "I paid $500 for a custom suit.",
        "I spent $120 on a high-end blender.",
        "I dropped $65 on a silk scarf.",
        "I shelled out $200 for a weekend at a hotel.",
        "I used $15 to buy a pair of socks.",
        "I spent $350 on a new couch.",
        "I bought a winter hat for $25.",
        "I paid $140 for a pair of boots.",
        "I spent $22 on a water bottle.",
        "I used $100 to buy a smartwatch strap.",
        "I dropped $75 on a gift for a friend.",
        "I just spent $60 on groceries.",
        "I paid $40 for a haircut.",
        "I bought a pet bed for $55.",
        "I spent $25 on a movie download.",
        "I used $18 to buy a novel.",
        "I purchased a pair of sunglasses for $200."
    ]
    date = datetime.now()
    for question in sentences:
        resp = chain.invoke({"question": question})
        print(resp)
        cur.execute("INSERT INTO expenses (price, description, category, date) VALUES (?, ?, ?, ?);", (resp.price, resp.description, resp.category, date))
        cur.fetchall()
