import pprint

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic.v1 import BaseModel, Field


class LocationDetail(BaseModel):
    """Extract all the details about a given recipe"""

    name: str = Field(description="The name of the recipe")
    # tags: list[str] = Field(description="The tags of the recipe, like vegan, no gluten, etc...")
    # ingredients: list[dict] = Field(description="The ingredients of the recipe")
    # instructions: list[dict] = Field(description="The instructions to prepare the recipe")
    # time: str = Field(description="The time needed to prepare the recipe")
    # difficulty: str = Field(description="The difficulty of the recipe")
    # rating: float = Field(description="The rating of the recipe")
    # price: str = Field(description="The price of the recipe")
    # description: str = Field(description="A description of the recipe")
    # calories: int = Field(description="The calories of the recipe")


model = OllamaFunctions(
    model="llama3.1",
    temperature=0.01,
    keep_alive=-1,
    format="json",
)

template = """You are a recipe extractor.
You will receive the content of a webpage, which contains a recipe, in the Context.
You must output the name, ingredients, instructions and description of 
the recipe, only with the information provided in the Context.
Context:
{context}."""


def get_chain(prompt):
    structured_llm = model.with_structured_output(LocationDetail)
    return prompt | structured_llm


def get_page_content(urls):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["a", "span"]
    )

    # Grab the first 3000 tokens of the site, in order to get only words and
    # meaningful stuff
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)

    return splits[2].page_content + splits[3].page_content + splits[4].page_content


def scrape_with_playwright(urls):
    content = get_page_content(urls)

    print(content)

    prompt = PromptTemplate.from_template(template)
    chain = get_chain(prompt)

    description = chain.invoke({"context": content})

    pprint.pprint(description)


if __name__ == '__main__':
    url = ["https://ricette.giallozafferano.it/Ratatouille.html"]
    scrape_with_playwright(url)
