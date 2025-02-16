import gradio as gr
import os
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from db_utils import DatabaseManager

# Initialize models and chains
llm = ChatOllama(model="llama3.1")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize text splitter with language-specific configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", "،", "؛", ":", " ", ""],
    keep_separator=True
)

# Initialize database, vector store and qa chain
db_manager = DatabaseManager()
vector_store = None
qa_chain = None
current_document_id = None

# Document loader mapping
document_loaders = {
    ".pdf": PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader
}

# Load the most recent document if it exists
documents = db_manager.get_documents()
if documents:
    try:
        most_recent_doc = documents[0]  # Get the most recent document
        current_document_id = most_recent_doc[0]  # ID is the first column
        file_path = most_recent_doc[2]  # Path is the third column
        
        # Reconstruct the vector store from the saved document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Recreate the QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
    except Exception as e:
        print(f"Error loading most recent document: {str(e)}")
        # Continue without the most recent document
        pass

def process_file(file):
    global vector_store, qa_chain, current_document_id
    
    try:
        # Get file extension and check if it's supported
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension not in document_loaders:
            return f"Unsupported file format: {file_extension}", None
        
        # Get appropriate loader
        loader_class = document_loaders[file_extension]
        loader = loader_class(file.name)
        documents = loader.load()
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        
        # Create or update vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Create QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        
        # Save document to database
        current_document_id = db_manager.add_document(os.path.basename(file.name), file.name)
        
        # Load existing chat history if any
        history = db_manager.get_chat_history(current_document_id)
        
        return "Document processed successfully! You can now start chatting.", history
    except Exception as e:
        return f"Error processing document: {str(e)}", None

def chat(message, history):
    if vector_store is None or current_document_id is None:
        return [], ""
    
    # Format history for the chain
    chat_history = []
    for msg in (history or []):
        chat_history.append((msg["content"] if msg["role"] == "user" else msg["content"], 
                           msg["content"] if msg["role"] == "assistant" else msg["content"]))
    
    # Get response from chain
    result = qa_chain({"question": message, "chat_history": chat_history})
    
    # Save messages to database
    db_manager.add_chat_message("user", message, current_document_id)
    db_manager.add_chat_message("assistant", result["answer"], current_document_id)
    
    # Return the chat history with the new message pair in the messages format
    if not history:
        history = []
    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["answer"]}
    ], ""

def load_document(doc_choice):
    global vector_store, qa_chain, current_document_id
    if not doc_choice:
        return None, None
    
    try:
        # Get document details from the selection
        doc_id, doc_name, file_path = doc_choice
        current_document_id = doc_id
        
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}", None
        
        # Get file extension and check if it's supported
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in document_loaders:
            return f"Unsupported file format: {file_extension}", None
        
        # Get appropriate loader
        loader_class = document_loaders[file_extension]
        loader = loader_class(file_path)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Recreate the QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        
        # Load chat history
        history = db_manager.get_chat_history(doc_id)
        return f"Loaded document: {doc_name}", history
    except Exception as e:
        return f"Error loading document: {str(e)}", None

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Chat with your Document")
    
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="Upload your Document", file_types=[".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"])
            process_button = gr.Button("Process document")
        with gr.Column(scale=1):
            # Get all documents from database
            all_docs = db_manager.get_documents()
            doc_choices = [(doc[1], doc) for doc in all_docs] if all_docs else []
            doc_dropdown = gr.Dropdown(
                choices=doc_choices,
                type="value",
                label="Select a document",
                value=doc_choices[0][1] if doc_choices else None
            )
            load_history_button = gr.Button("Load Chat History")
    
    with gr.Row():
        status_text = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.Chatbot(label="Chat History", type="messages")
    with gr.Row():
        msg = gr.Textbox(label="Your Message")
        submit = gr.Button("Submit")
    clear = gr.Button("Clear")

    # Set up event handlers
    process_button.click(process_file, inputs=[file_input], outputs=[status_text, chatbot])
    doc_dropdown.change(load_document, inputs=[doc_dropdown], outputs=[status_text, chatbot])
    load_history_button.click(load_document, inputs=[doc_dropdown], outputs=[status_text, chatbot])
    msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    submit.click(chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear.click(lambda: [], None, chatbot, queue=False)

demo.launch(share=True)