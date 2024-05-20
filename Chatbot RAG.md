

# Workflow
![[Screenshot 2024-04-30 alle 12.32.51.png]]

# Doc loading
```python
from langchain_community.document_loaders import TextLoader  
  
loader = TextLoader("./text_file.md")  
doc = loader.load()
```
# Chunking
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter  
  
text_splitter = RecursiveCharacterTextSplitter(  
# Set a really small chunk size, just to show.  
chunk_size=100,  
chunk_overlap=20,  
length_function=len,  
is_separator_regex=False,  
)  
  
texts = text_splitter.create_documents(doc)
```
# Embedding
```python
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
```
# Vector storing
```python
from langchain.vectorstores import FAISS

db = FAISS.from_texts(texts, embeddings)
```
# Retrieving
```python
retriever = db.as_retriever()
```
# LLM
```python
from langchain_community.llms import HuggingFaceHub

llm = HuggingFaceHub(  
	repo_id="model",  
	task="text-generation",  
	model_kwargs={  
		"max_new_tokens": 512,  
		"top_k": 30,  
		"temperature": 0.1,  
		"repetition_penalty": 1.03,  
	},  
	huggingfacehub_api_token= hg_key,
)
```
# Template
```python
template = """Answer the question based only on the following context:  
  
{context}  
  
Question: {question}  
"""  

prompt = ChatPromptTemplate.from_template(template)
```
# Bot
```python
from langchain_core.runnables import RunnablePassthrough

def chat_with_rag(message):
	def format_docs(docs):  
		return "\n\n".join([d.page_content for d in docs])

	chain = (  
		{"context": retriever | format_docs, "question": RunnablePassthrough()}  
		| prompt  
		| llm  
		| StrOutputParser()  
	)  
  
	return chain.invoke(message)
```
dove posso trovare la sezione servizi?