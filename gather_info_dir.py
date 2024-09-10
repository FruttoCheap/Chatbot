import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

website = ""

persist_directory = "./chroma/chroma_sys"

text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)

embeddings = HuggingFaceEmbeddings()


def pdf_to_text(file_path):
    import PyPDF2
    # Open the PDF file in read-binary mode
    with open(file_path, 'rb') as file:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Get the number of pages in the PDF file
        num_pages = len(pdf_reader.pages)

        # Loop through each page and extract the text
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            embed(file_path + " - " + f"{page_num}", text)


office_extensions = [".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".odt", ".ods", ".odp"]


def office_to_text(file_path):
    from tika import parser
    raw = parser.from_file(file_path)
    return raw['content']


def html_to_text(html):
    from bs4 import BeautifulSoup
    with open(html, 'r') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()


def csv_to_text(file_path):
    import csv
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        text = ""
        for row in reader:
            text += str(row) + "\n"


def json_to_text(file_path):
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
        text = ""
        for key, value in data.items():
            text += f"{key}: {value}\n"
        return text


def xml_to_text(file_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_path)
    root = tree.getroot()
    text = ""
    for child in root:
        text += f"{child.tag}: {child.text}\n"
    return text


def md_to_text(md):
    import markdown
    from bs4 import BeautifulSoup
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


def yml_to_text(file_path):
    import yaml
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def toml_to_text(file_path):
    import toml
    with open(file_path, 'r') as file:
        return toml.load(file)


# still to check
def ini_to_text(file_path):
    import configparser
    config = configparser.ConfigParser()
    config.read(file_path)
    return {s: dict(config.items(s)) for s in config.sections()}


def properties_to_text(file_path):
    import configparser
    config = configparser.ConfigParser()
    with open(file_path, 'r') as f:
        file_content = '[section]\n' + f.read()
    config.read_string(file_content)
    return dict(config['section'])


def ipynb_to_text(file_path):
    import nbformat
    with open(file_path, 'r') as file:
        return nbformat.read(file, as_version=4)


def db_to_text(file_path):
    import sqlite3
    con = sqlite3.connect(file_path)
    cur = con.cursor()

    # get tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()

    # get data for each table
    for table in tables:
        text = table[0] + "\n"
        cur.execute(f"SELECT * FROM {table[0]}")
        rows = cur.fetchall()
        for row in rows:
            text += str(row) + "\n"
        embed(table[0], text)


plain_text_extensions = [".py", ".js", ".jsx", ".css", ".c", ".cpp", ".h", ".java", ".php", ".rb", ".pl", ".sh", ".bat",
                         ".ps1", ".sql", ".cfg", ".conf", ".env", ".rst", ".ts", ".tsx", ".go", ".rs", ".swift", ".kt",
                         ".kts", ".dart", ".scala", ".groovy", ".log", ".txt", ".rtf"]


def plain_to_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def collect_files(file):
    entries = [file + "/" + f for f in os.listdir(file)]
    print(entries)
    if not entries:
        return

    if entries:
        for f in entries:
            if f.startswith("."):
                continue
            # document formats
            elif f.endswith(".pdf"):
                pdf_to_text(f)
            elif os.path.splitext(f)[1] in office_extensions:
                embed(f, office_to_text(f))
            elif f.endswith(".csv"):
                embed(f, csv_to_text(f))
            elif f.endswith(".md"):
                embed(f, md_to_text(f))
            # code formats
            elif os.path.splitext(f)[1] in plain_text_extensions:
                embed(f, plain_to_text(f))
            elif f.endswith(".html"):
                embed(f, html_to_text(f))
            elif f.endswith(".json"):
                embed(f, json_to_text(f))
            elif f.endswith(".xml"):
                embed(f, xml_to_text(f))
            elif f.endswith(".yaml") or f.endswith(".yml"):
                embed(f, yml_to_text(f))
            elif f.endswith(".toml"):
                embed(f, toml_to_text(f))
            elif f.endswith(".ini"):
                embed(f, ini_to_text(f))
            elif f.endswith(".properties"):
                embed(f, properties_to_text(f))
            elif f.endswith(".ipynb"):
                embed(f, ipynb_to_text(f))
            # database formats
            elif f.endswith(".db") or f.endswith(".sqlite3"):
                db_to_text(f)

    dirs = [d for d in entries if os.path.isdir(d)]

    if dirs:
        for d in dirs:
            collect_files(d)


def embed(filename, text):
    texts = text_splitter.split_text(text)
    if texts:
        vectordb = Chroma.from_texts([t for t in texts], embeddings, persist_directory=persist_directory,
                                     metadatas=[{"file": filename}])
        vectordb.persist()


if __name__ == '__main__':
    os.listdir(".")
    dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
    path = input("Available directories: " + str(dirs) + "\nEnter directory: ")
    collect_files(path)
