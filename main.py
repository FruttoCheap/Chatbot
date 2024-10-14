from os import environ, getenv
from sqlite3 import connect
from sys import exit
from dotenv import load_dotenv
from module_NLP import get_database, get_NLP_chains, NLP
from module_RAG import get_embedded_database, RAG, stripOutput
from module_choose_NLP_RAG_input_plot import get_tagging_chain, get_classification
from module_input import get_input_chain, input_into_database
from module_plot_SVG import get_plot_model, get_plot_from_all, get_plot_from_RAG
from module_chartJS import (
    get_data_chain, get_labels_chain, get_type_chain,
    get_graph_type, get_labels, get_data_RAG, get_data_NLP,
    write_chart_html, get_label_title, get_label_chain, get_chart_description_chain
)

# Constants for configuration
URI_DB = "sqlite:///googleDb.sqlite3"
PERSIST_DIRECTORY = "./chroma/expenses"
INPUT_DB = "googleDb.sqlite3"
MAX_DESCRIPTION_LENGTH = 255

PRINT_SETTINGS = {
    "print_question": False,
    "print_query": False,
    "print_description": False,
    "print_corrected_query": False,
    "print_time": False,
    "print_scores": False,
    "print_chunks": False,
    "print_smallest_chunk": False,
    "print_context": False,
    "print_method": False,
    "print_characteristics_of_the_question": False,
    "print_explaination_plot": False,
    "call_SVG_plot": False,
    "call_JSON_plot": True,
    "print_plot_type": False,
    "print_plot_labels": False,
    "print_plot_data": False
}

# Load environment variables from .env file
def load_environment_variables():
    load_dotenv()
    environ["GROQ_API_KEY"] = getenv("GROQ_API_KEY")

# Initialize NLP components
def initialize_nlp(uri_db):
    nlp_db = get_database(uri_db)
    full_chain, correction_chain, description_chain = get_NLP_chains(nlp_db)
    return nlp_db, full_chain, correction_chain, description_chain

# Initialize RAG database
def initialize_rag(persist_directory):
    return get_embedded_database(persist_directory)

# Handle SVG plot generation
def handle_svg_plot(method, question, connection, rag_db, PRINT_SETTINGS):
    if method[0] == "NLP":
        get_plot_from_all(get_plot_model(), connection, question, PRINT_SETTINGS)
    elif method[0] == "RAG":
        response = RAG(question, rag_db, stripOutput, PRINT_SETTINGS, is_for_plot=True)
        get_plot_from_RAG(get_plot_model(), response, question, PRINT_SETTINGS)

# Handle JSON plot generation
def handle_json_plot(method, question, labels_chain, label_chain, type_chain, data_chain, rag_db, nlp_db, chart_description_chain, PRINT_SETTINGS):
    label = get_label_title(question, label_chain)
    
    if method[0] == "NLP":
        response = NLP(question, nlp_db, *initialize_nlp(URI_DB)[1:], PRINT_SETTINGS)
        chart_type = get_graph_type(type_chain, question)
        labels = get_labels(labels_chain, question, chart_type, response, PRINT_SETTINGS)
        try:
            data = get_data_NLP(labels, response, PRINT_SETTINGS)
        except Exception:
            print("There was an error in the creation of the chart. Try to be more specific.")
            return
    elif method[0] == "RAG":
        response = RAG(question, rag_db, stripOutput, PRINT_SETTINGS, is_for_plot=True)
        chart_type = get_graph_type(type_chain, question)
        labels = get_labels(labels_chain, question, chart_type, response, PRINT_SETTINGS)
        try:
            data = get_data_RAG(data_chain, question, chart_type, labels, response, PRINT_SETTINGS)
        except Exception:
            print("There was an error in the creation of the chart. Try to be more specific.")
            return

    if not any(item != 0.0 for item in data):
        print("There was an error in the creation of the chart. Try to be more specific.")
    else: 
        if len(labels) == 1:
            chart_type = "bar"
        if (PRINT_SETTINGS["print_plot_type"]):
            print(f"Chart type: {chart_type}")
        
        write_chart_html(chart_type, labels, data, label, chart_description_chain)

# Main function to run the chatbot
def main():
    load_environment_variables()

    # Get the method chain
    tagging_chain = get_tagging_chain()

    # Initialize input database
    connection = connect(INPUT_DB)
    cursor = connection.cursor()
    input_chain = get_input_chain()

    # Initialize NLP and RAG
    nlp_db, full_chain, correction_chain, description_chain = initialize_nlp(URI_DB)
    rag_db = initialize_rag(PERSIST_DIRECTORY)

    # Initialize plot chains
    label_chain = get_label_chain()
    type_chain = get_type_chain()
    data_chain = get_data_chain()
    labels_chain = get_labels_chain()
    chart_descr_chain = get_chart_description_chain()

    while True:
        question = input("Enter your question ('X' or 'x' to exit): ")
        if question.lower() == 'x':
            exit()
        method = get_classification(question, tagging_chain, PRINT_SETTINGS)
        if PRINT_SETTINGS["print_method"]:
                print(f"Chosen method: {method}.")
        if method == "REJECTED":
            response = "The question is not related to personal finance"
        elif method == "rejected (exception)":
            response = "An error occurred. Try again."
        elif method == "INPUT":
            response = input_into_database(question, input_chain, cursor, MAX_DESCRIPTION_LENGTH)
        elif "|" not in method:
            if method == "NLP":
                response = NLP(question, nlp_db, full_chain, correction_chain, description_chain, PRINT_SETTINGS)
            else:
                response = RAG(question, rag_db, stripOutput, PRINT_SETTINGS)
        elif PRINT_SETTINGS["call_JSON_plot"]:
            handle_json_plot(method.split("|"), question, labels_chain, label_chain, type_chain, data_chain, rag_db, nlp_db, chart_descr_chain, PRINT_SETTINGS)
        elif PRINT_SETTINGS["call_SVG_plot"]:
            handle_svg_plot(method.split("|"), question, connection, rag_db, PRINT_SETTINGS)
        else:
            response = "An error occurred while trying to classify the question. Try again."
        if "PLT" not in method:
                print(f"Response: {response}.")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"An error occurred: {error}.")
        exit(1)