import os
import sqlite3
from dotenv import load_dotenv
from module_NLP import get_database, get_NLP_chains, NLP
from module_RAG import get_embedded_database, RAG, stripOutput
from module_choose_NLP_RAG_input_plot import get_tagging_chain, get_classification
from module_input import get_input_chain, input_into_database
from module_plot import get_plot_model, get_plot_from_all, get_plot_from_RAG

# Constants for configuration
URI_DB = "sqlite:///googleDb.sqlite3"
PERSIST_DIRECTORY = "./chroma/expenses"
INPUT_DB = "googleDb.sqlite3"
MAX_DESCRIPTION_LENGTH = 255
PRINT_SETTINGS = {
    "print_question": False,
    "print_query": False,
    "print_description": True,
    "print_corrected_query": False,
    "print_time": False,
    "print_scores": False,
    "print_chunks": False,
    "print_smallest_chunk": True,
    "print_context": True,
    "print_method": True,
    "print_characteristics_of_the_question": False,
    "print_explaination_plot": True
}

# definition of databases and important variables
def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def initialize_nlp(uri_db):
    """Initialize NLP chains and database."""
    nlp_db = get_database(uri_db)
    full_chain, correction_chain, description_chain = get_NLP_chains(nlp_db)
    return nlp_db, full_chain, correction_chain, description_chain

def initialize_rag(persist_directory):
    """Initialize RAG database."""
    return get_embedded_database(persist_directory)

# main function: it will loop indefinitely on the questions of the user.
def main():
    """Main function to run the chatbot."""

    load_environment_variables()

    # Get the classification chain
    tagging_chain = get_tagging_chain()

    # Initialize INPUT
    connection = sqlite3.connect(INPUT_DB)
    cursor = connection.cursor()
    input_chain = get_input_chain()
    
    # Initialize NLP and RAG
    nlp_db, full_chain, correction_chain, description_chain = initialize_nlp(URI_DB)
    rag_db = initialize_rag(PERSIST_DIRECTORY)

    while True:
        # Get the question
        question = input("Enter your question ('X' or 'x' to exit): ")

        # Check if the user wants to exit
        if question == 'X' or question == 'x':
            exit()

        # Choose between RAG, NLP or reject the question
        method = get_classification(question, tagging_chain, PRINT_SETTINGS)

        # Debug: print the chosen method
        if PRINT_SETTINGS["print_method"]:
            print(f"Chosen method: {method}.")

        # Run the chosen method
        if method == "REJECTED":
            response = "The question is not related to personal finance"
        elif method == "rejected (exception)":
            response = "An error occurred. Try again"
        elif method == "INPUT":
            response = input_into_database(question, input_chain, cursor, MAX_DESCRIPTION_LENGTH)
        elif "|" not in method:
            if method == "NLP":
                response = NLP(question, nlp_db, full_chain, correction_chain, description_chain, PRINT_SETTINGS)
            elif method == "RAG":
                response = RAG(question, rag_db, stripOutput, PRINT_SETTINGS)
        elif "|" in method:
            method = method.split("|")
            if method[0] == "NLP":
                get_plot_from_all(get_plot_model(), connection, question, PRINT_SETTINGS)
            elif method[0] == "RAG":
                response = RAG(question, rag_db, stripOutput, PRINT_SETTINGS, is_for_plot=True)
                print(response)
                get_plot_from_RAG(get_plot_model(), response, question, PRINT_SETTINGS)
        else:
            response = "An error occurred while trying to classify the question. Try again"

        # Print the response
        if ("PLT" not in method):
            print(f"Response: {response}.")

if __name__ == "__main__":
    main()