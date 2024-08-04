import pprint

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic.v1 import BaseModel, Field


class RecipeDetail(BaseModel):
    """Extract all the details about a given recipe"""

    name: str = Field(description="The name of the recipe")


model = OllamaFunctions(
    model="llama3.1",
    temperature=0.01,
    keep_alive=-1
)

template = """You are a recipe extractor.
You will receive the content of a webpage, which contains a recipe, in the Context.
You must output the name of the recipe using only the information provided in the Context.
Context:
{context}."""


def get_chain(prompt):
    structured_llm = model.with_structured_output(RecipeDetail)
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

    prompt = PromptTemplate.from_template(template)
    chain = get_chain(prompt)

    description = chain.invoke({"context": content})

    # pprint.pprint(description)
    print(description)


if __name__ == '__main__':
    url = ["https://ricette.giallozafferano.it/Ratatouille.html"]
    scrape_with_playwright(url)
