from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize the ChatOllama model
llm = ChatOllama(model="llama3.2-vision")

# Generate a response
response = llm.invoke("Hello, how are you?")
#print(response)

# Load PDF
loader = PyPDFLoader("data/SeniorEmirati.pdf")
documents = loader.load()   




# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)




# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Remove the manual embedding step as FAISS.from_documents handles it
# Create FAISS vector store
vector_store = FAISS.from_documents(chunks, embeddings)




# Define your query
query = "Summerize the uploaded paper"

# Perform similarity search
relevant_chunks = vector_store.similarity_search(query)

# Display the most relevant chunk
print(relevant_chunks[0].page_content)
