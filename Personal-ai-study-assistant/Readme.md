# Personal AI Study Assistant

## Project Overview
The **Personal AI Study Assistant** is an advanced chatbot that helps users analyze and summarize research papers. It leverages a **Retrieval-Augmented Generation (RAG)** pipeline with NLP techniques, vector databases, and a large language model (LLM) to provide insightful responses based on uploaded research papers.

## Features
- **Automatic PDF Parsing**: Extracts text, metadata (title, authors), and relevant sections from research papers.
- **Named Entity Recognition (NER)**: Identifies authors and titles using SpaCy’s NLP model.
- **Chunking for Efficient Retrieval**: Splits documents into smaller text chunks for optimized query response.
- **Embedding Model for Semantic Search**: Uses `HuggingFaceEmbeddings` to create vector representations of text.
- **Vector Database for Fast Retrieval**: Stores and retrieves document chunks using `ChromaDB`.
- **LLM-Powered Responses**: Uses `ChatGroq` (Mixtral-8x7b-32768) to answer user queries based on research content.
- **Streamlit-based UI**: A simple and interactive front-end built with Streamlit to facilitate seamless user interaction.
- **Query Logging and Tracking**: Logs user queries and responses for further analysis and improvement.

## Setup Instructions
### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/personal-ai-study-assistant.git
cd personal-ai-study-assistant
```

### 2. Install Dependencies
Ensure you have Python 3.8+ and install the required libraries:
```sh
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root with:
```
GROQ_API_KEY=your_api_key_here
```

### 4. Organize Input PDFs
Place research papers in the `input` folder:
```
./input/
  ├── paper1.pdf
  ├── paper2.pdf
```

### 5. Run the Application
```sh
python main.py
```

## Chosen Technologies
- **Chunking Method**: `RecursiveCharacterTextSplitter` with `chunk_size=500`, `chunk_overlap=50`.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face).
- **Vector Database**: `ChromaDB`, storing embeddings for efficient retrieval.
- **LLM for Querying**: `ChatGroq (Mixtral-8x7b-32768)` for intelligent research-based answers.
- **UI Framework**: `Streamlit` for an easy-to-use interface.

## Usage
- The application extracts and processes research papers into a vector database.
- Users can input queries related to the research papers.
- The chatbot responds with context-aware summaries and insights.
- Responses are logged for future reference and improvement.

## Project Demo
A demo video showcasing the chatbot’s functionality, document retrieval, and AI responses will be uploaded to YouTube.
Research Assistant Demo:
[YouTube Demo](https://youtu.be/A0adN2_fbMU)

## Queries & Support
For any queries, reach out via [GitHub Issues](https://github.com/yourusername/personal-ai-study-assistant/issues).

