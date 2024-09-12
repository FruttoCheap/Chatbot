from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from pydantic import BaseModel, Field


class Answer(BaseModel):
    """Outputs the structured version of the user input."""

    price: float = Field(description="The sum of money spent by the user")
    description: str = Field(description="A short description of the user's expense")
    category: str = Field(description="The category of the user's expense")


model = OllamaFunctions(
    model="google",
    keep_alive=-1,
    format="json",
)

def get_prompt():
    template = """UserInput: {question}."""
    prompt = PromptTemplate.from_template(template)
    return prompt


def get_chain(prompt):
    structured_llm = model.with_structured_output(Answer)
    return prompt | structured_llm


def get_answer(q):
    chain = get_chain(get_prompt())
    result = chain.invoke({"question": q})
    print(result)


if __name__ == '__main__':
    query = input("Query: ")
    get_answer(query)
