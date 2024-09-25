import os
from dotenv import load_dotenv
from module_NLP import get_database, get_NLP_chains, NLP
from module_RAG import get_embedded_database, RAG, stripOutput
from module_choose_NLP_RAG import get_tagging_chain, get_classification

# Constants for configuration
URI_DB = "sqlite:///googleDb.sqlite3"
PERSIST_DIRECTORY = "./chroma/expenses"
PRINT_SETTINGS = {
    "print_question": False,
    "print_query": True,
    "print_description": True,
    "print_corrected_query": True,
    "print_time": False,
    "print_scores": False,
    "print_chunks": False,
    "print_smallest_chunk": False,
    "print_context": False,
    "print_method": True,
    "print_characteristics_of_the_question": False
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
    k_size = int(nlp_db.run("SELECT COUNT(*) from expensesok;").replace("[(","").replace(",)]",""))
    return nlp_db, full_chain, correction_chain, description_chain, k_size

def initialize_rag(persist_directory):
    """Initialize RAG database."""
    return get_embedded_database(persist_directory)

# main function: it will loop indefinitely on the questions of the user.
def main():
    """Main function to run the chatbot."""
    load_environment_variables()

    # Get the classification chain
    tagging_chain = get_tagging_chain()

    # Initialize NLP and RAG
    nlp_db, full_chain, correction_chain, description_chain, k_size = initialize_nlp(URI_DB)
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
        if method == "rejected":
            response = "Not a relevant question"
        elif method == "rejected (exception)":
            response = "An error occurred. Try again"
        elif method == "NLP":
            response = NLP(question, nlp_db, full_chain, correction_chain, description_chain, PRINT_SETTINGS)
        elif method == "RAG":
            response = RAG(question, rag_db, stripOutput, PRINT_SETTINGS, k_size)

        # Print the response
        print(f"Response: {response}.")

if __name__ == "__main__":
    main()