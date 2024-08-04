import pprint

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

response_schemas = [
    ResponseSchema(name="name", description="name of the location asked by the user"),
    ResponseSchema(
        name="short_description",
        description="a short description of the location asked by the user",
    ),
    ResponseSchema(
        name="extended_description",
        description="a long description of the location asked by the user",
    ),
]


def scrape_with_playwright(urls):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["a", "span"]
    )

    # Grab the first 3000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)

    print("Extracting content with LLM")

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="Find as many info as possible about the user input using the provided context.\n{"
                 "format_instructions}\n{context}\n{question}",
        input_variables=["question", "context"],
        partial_variables={"format_instructions": format_instructions},
    )

    model = ChatOllama(model="llama3.1", temperature=0.01)
    chain = prompt | model

    description = chain.invoke({"context": splits[0].page_content, "question": "Farmacia del cambio"})

    print(description.content)

    prompt = PromptTemplate(
        template="Find as many info as possible about the location provided.\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )

    parsing_chain = prompt | model | output_parser

    parsed = parsing_chain.invoke({"question": description.content})
    # parsed = parsing_chain.invoke(splits[0].page_content)

    pprint.pprint(parsed)


if __name__ == '__main__':
    urls = ["https://www.google.com/search?q=farmacia+del+cambio"]
    scrape_with_playwright(urls)
