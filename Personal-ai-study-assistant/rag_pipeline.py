import os
import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import logging
import re
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# Set up logging to log information to a file
log_file =  os.getenv("LOG_FILE_PATH")
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

class RAGPipeline:
    def __init__(self, folder_path="input", db_path="chroma_db", groq_model="mixtral-8x7b-32768"):
        """Initialize paths, embedding model, vector database, and ChatGroq model."""
        self.folder_path = folder_path
        self.db_path = db_path
        self.chat_model = ChatGroq(model=groq_model, api_key=api_key)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="research_papers")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.papers_info = []
        self.selected_doc = None
        self.context = None
        self.final_prompt = None

    def process_pdfs(self, specific_files=None):
        """Extracts text from PDFs, creates embeddings, and stores in ChromaDB."""
        files_to_process = specific_files if specific_files else os.listdir(self.folder_path)
        for file in files_to_process:
            if file.endswith(".pdf"):
                file_path = os.path.join(self.folder_path, file)
                loader = PyMuPDFLoader(file_path)
                pages = loader.load()

                # Extract metadata (title & authors)
                extracted_text = "\n".join([p.page_content for p in pages])
                title, authors = self.extract_metadata(extracted_text)

                # Split text into chunks & store in ChromaDB
                chunks = self.text_splitter.split_text(extracted_text)
                for idx, chunk in enumerate(chunks):
                    self.collection.add(
                        documents=[chunk],
                        metadatas={"title": title, "authors": authors, "page": idx + 1, "file": file},
                        ids=[f"{file}_{idx}"]
                    )
                # Store paper info for listing
                self.papers_info.append({
                    'title': title,
                    'filename': file,
                    'authors': authors
                })
        return self.papers_info


    @staticmethod
    def extract_metadata(text):
        """Simple heuristic to extract title and authors from the first few lines."""
        lines = text.split("\n")
        title = lines[0].strip() if len(lines) > 0 else "Unknown Title"
        authors = lines[1].strip() if len(lines) > 1 else "Unknown Authors"
        return title, authors

    def list_papers(self):
        """Returns a formatted string of available papers."""
        if not self.papers_info:
            return "❌ No research papers found. Please upload PDFs to the 'pdfs' folder."

        paper_list = "\n".join([f"{idx + 1}. {paper['title']} ({paper['filename']})" for idx, paper in enumerate(self.papers_info)])
        return f"📚 **Available Research Papers:**\n{paper_list}\n\nEnter the document number to select."

    def query_documents(self, query):
        """Retrieves relevant document chunks and uses ChatGroq for response articulation."""
        
        # Define patterns for commands that indicate the user wants to list documents or go to the main menu
        menu_keywords = [
            r"(list|show)\s*(all)?\s*(documents|papers)",   # matches "list all documents", "show papers"
            r"(go|return)\s*to\s*(main\s*menu|document\s*list)",  # matches "go to main menu", "return to documents list"
            r"(documents|papers)\s*(list|menu)",  # matches "documents list", "papers menu"
            r"(list|menu)",  # matches "list", "menu"
        ]
        
        # Define patterns for elaboration requests
        elaboration_keywords = [
            r"(elaborate|explain|expand|more details?)\s*(on)?\s*(.*)",  # matches "elaborate", "explain", "expand", "more details"
        ]
        
        # Compile the regular expressions for efficiency
        menu_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in menu_keywords]
        elaboration_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in elaboration_keywords]

        # Check if any of the patterns match the user's query
        if any(pattern.search(query) for pattern in menu_patterns):
            return self.list_papers()  # Return the list of papers again
        
       
        if not self.selected_doc:
            return "❌ No document selected. Please choose a document before asking queries."
        
        # Retrieve the top 3 most relevant document chunks for the selected paper
        similar_docs = self.collection.query(
            query_texts=[query],
            n_results=3
        )

        if not similar_docs["documents"][0]:
            return "❌ No relevant information found in the selected document."

        # Extract relevant context
        context_parts = []
        for i, doc in enumerate(similar_docs["documents"][0]):
            metadata = similar_docs["metadatas"][0][i]
            file = metadata.get("file", "")

            # Ensure response is only from the selected document
            if file == self.selected_doc:
                title = metadata.get("title", "Unknown Document")
                page = metadata.get("page", "N/A")
                context_parts.append(f"📄 **{title}** (Page {page}):\n{doc}")

        if not context_parts:
            return "❌ No matching content found in the selected document."

        # Combine retrieved chunks as context
        context = "\n\n".join(context_parts)

        # Check if it's a follow-up query (related to the previous context)
        if self.context and query.lower() in self.context.lower():
            final_prompt = f"Elaborate on the previous answer:\n\n{self.context}\n\n{query}"
        else:
            final_prompt = f"""
You are an expert AI assistant trained on the given document excerpts. Your goal is to provide a well-structured, clear, and informative response that is easy to understand for everyone. Please follow the instructions below:

Instructions:
- First-time interaction: Greet the user warmly, introduce yourself, and kindly request them to select a document from the list.
- If the user greets: Respond with a friendly greeting, provide a brief introduction about yourself, and again ask them to select a document from the list.
- Think step-by-step to fully understand the query.
- Break the query down into sub-questions if needed.
- Focus on expanding the information related to the selected document.
- Retrieve context from the user's query for a better response.
- Use only the provided document excerpts for your response.
- If information is missing, state that clearly instead of guessing.
- For follow-up queries, check if the question relates to previous context. If yes, build on it; otherwise, answer independently based on the provided excerpts.
- Provide a detailed answer or elaboration when asked.
- If the user requests a summary, provide a concise and clear summary of the document excerpts, highlighting the key takeaways.
- If the user asks for more details, elaborate on the content already provided, offering deeper insights, examples, and clarifications.
- Use bullet points, analogies, or examples to make the information easy to follow.


    ### **Document Excerpts:**  
    {context}  

    ---

    ### **User Query:**  
    {query}  

    ### **Final Answer:**  
    """
        # Use ChatGroq to generate a refined answer
        messages = [
            HumanMessage(content=final_prompt)
        ]
        
        response = self.chat_model.invoke(messages)

        # Log the response from the model
        logging.info(f"Document Name: {self.selected_doc}")
        logging.info(f"User Query: {query}")
        logging.info(f"Context: {context}")
        logging.info(f"Final Prompt: {final_prompt}")
        logging.info(f"Model Response: {response.content if response else 'No response generated.'}")
        
        logging.debug(f"Model Response: {response.content}")

        # Save the context for follow-up queries
        self.context = context

        return response.content if response else "❌ ChatGroq failed to generate a response."


    

    def select_document(self, doc_index):
        """Sets the selected document for queries."""
        if 0 <= doc_index < len(self.papers_info):
            self.selected_doc = self.papers_info[doc_index]['filename']
            return f"✅ You selected: **{self.papers_info[doc_index]['title']}**\nNow, ask your query related to this document!"
        return "❌ Invalid selection. Please choose a valid document number."
