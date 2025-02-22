# Code Explanation - Personal AI Study Assistant

## 1. Initialization (`__init__` Method)
The `__init__` method initializes the **Retrieval-Augmented Generation (RAG) pipeline** by setting up core components, including document storage, embeddings, and the AI model for response generation.

- **`folder_path`**: Defines the directory where input PDF files are stored. The default value is `input/`.
- **`db_path`**: Specifies the location of the ChromaDB vector store (default: `chroma_db`), used for efficient document retrieval.
- **`groq_model`**: Sets the AI model (`mixtral-8x7b-32768`) used for generating responses via the Groq API.
- **`chat_model`**: Establishes a connection to ChatGroq for handling user queries.
- **`client`**: Initializes a connection to **ChromaDB**, the vector database used for storing document embeddings.
- **`collection`**: Defines the ChromaDB collection where documents and their embeddings are stored.
- **`embeddings`**: Utilizes the `HuggingFaceEmbeddings` model to convert text into vector representations for efficient similarity searches.
- **`text_splitter`**: Implements `RecursiveCharacterTextSplitter` to divide large documents into smaller chunks for improved retrieval.
- **`papers_info`**: Stores metadata (title, authors, filename) of processed research papers for easy reference.
- **`selected_doc`**: Tracks the currently selected document for user queries.
- **`context` & `final_prompt`**: Maintain the query’s contextual information and the final structured prompt sent to ChatGroq for response generation.

---

## 2. Processing PDFs (`process_pdfs` Method)
This method extracts and processes research papers, converting them into searchable document embeddings.

- **Text Extraction**: Utilizes `PyMuPDFLoader` to extract raw text from each page of the PDF.
- **Metadata Extraction**: Extracts **title and author names** from the document's first few lines (heuristic-based approach).
- **Text Chunking**: Uses `RecursiveCharacterTextSplitter` to break down extracted text into **smaller, meaningful segments** to improve retrieval efficiency.
- **Storage in ChromaDB**: Each chunk is stored in **ChromaDB**, along with metadata (title, authors, and page numbers), to enable efficient semantic search.
- **Paper Info Storage**: Updates `self.papers_info` with document metadata for listing and user selection.

---

## 3. Listing Available Papers (`list_papers` Method)
This method returns a list of processed papers, allowing users to view available research documents.

- If **papers exist**, it returns their titles and filenames.
- If **no papers are found**, it prompts the user to upload PDFs before proceeding.

---

## 4. Querying Documents (`query_documents` Method)
Handles user queries, retrieves relevant document excerpts, and generates AI-driven responses.

### Key Steps:
1. **User Input Handling**: Checks if the query matches predefined commands (`list papers`, `go back to menu`, etc.) using regular expressions.
2. **Document Selection Enforcement**: If no document is selected, prompts the user to choose one before querying.
3. **Document Retrieval**:
   - Searches the **ChromaDB collection** for relevant document chunks matching the user’s query.
   - If no relevant excerpts are found, the system informs the user.
4. **Contextualized Response Generation**:
   - Compiles the retrieved text chunks to form a **contextual foundation** for the response.
   - Sends the query and relevant context to **ChatGroq** for generating a well-structured answer.
5. **Follow-up Query Handling**:
   - If a user follows up on a previous response, the system builds upon the existing context for improved coherence.
6. **ChatGroq Invocation**:
   - Submits the structured query to ChatGroq, which generates an **accurate and well-formatted response** in Markdown.

---

## 5. Document Selection (`select_document` Method)
Allows users to choose a research paper from the available list before querying.

### Workflow:
1. **Processes and stores** PDFs in ChromaDB.
2. **User selects a document** from the available papers.
3. **Queries are executed** based on the selected document.
4. **Relevant excerpts are retrieved** from ChromaDB.
5. **ChatGroq generates AI-driven responses** based on the extracted document context.

---

## 6. User Interaction (`main.py`)
The **main entry point** for the application, where users interact with the AI study assistant.

### Initialization:
- Instantiates the **RAGPipeline**.
- Automatically **processes PDFs** stored in the `input/` directory.

### Interaction Flow:
1. **Users select a document** or **issue a query**.
2. If a **document is selected**, users can ask **contextual questions** about it.
3. The system retrieves **relevant document excerpts** and generates AI-powered responses.
4. Users can ask **follow-up questions** to refine their understanding.
5. Typing `exit` terminates the session.

### Key Features:
- **Document Retrieval**: Efficiently fetches relevant excerpts from research papers.
- **Contextual Answer Generation**: Provides AI-driven responses with accurate context.
- **User-Friendly Interaction**: Ensures users select a document before querying, maintaining a structured workflow.

---

## Conclusion
The **Personal AI Study Assistant** leverages **NLP, vector databases, and AI models** to enable efficient research paper analysis. This pipeline ensures **fast retrieval, contextualized responses, and a seamless user experience**, making it an essential tool for researchers and students.