# Chat with your Documents

A powerful document interaction application that enables natural language conversations with your documents using LangChain and Ollama. This application supports multiple document formats and maintains persistent chat history for seamless document interactions.

## 🌟 Features

- **Multi-Format Document Support**
  - PDF files (.pdf)
  - Word documents (.doc, .docx)
  - PowerPoint presentations (.ppt, .pptx)
  - Excel spreadsheets (.xls, .xlsx)

- **Intelligent Document Processing**
  - Advanced text chunking with language-aware splitting
  - Efficient vector storage using FAISS
  - Multilingual support with paraphrase-multilingual-MiniLM-L12-v2 embeddings

- **Interactive Chat Interface**
  - User-friendly Gradio web interface
  - Persistent chat history across sessions
  - Context-aware responses using ConversationalRetrievalChain

- **Document Management**
  - Easy document upload and processing
  - Quick switching between documents
  - Historical chat retrieval for each document

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Ollama installed and running locally
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ChatWithYourDoc
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Start the application:
```bash
python gradio_app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:7860)

3. Upload a document or select from previously uploaded documents

4. Start chatting with your document!

## 🏗️ Architecture

### Components

- **Document Processing**
  - Uses LangChain's document loaders for multiple file formats
  - Implements RecursiveCharacterTextSplitter for intelligent text chunking
  - Employs FAISS for efficient vector similarity search

- **Language Model Integration**
  - Integrates with Ollama for local LLM inference
  - Utilizes HuggingFace embeddings for document vectorization
  - Implements ConversationalRetrievalChain for context-aware responses

- **Database Management**
  - SQLite database for persistent storage
  - Stores document metadata and chat history
  - Enables seamless chat history retrieval

### Tech Stack

- **Core Framework**: LangChain
- **UI Framework**: Gradio
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace Transformers
- **Database**: SQLite
- **LLM**: Ollama (llama3.1)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or feedback, please open an issue in the repository.
