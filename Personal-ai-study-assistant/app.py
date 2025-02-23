import streamlit as st
import os
from rag_pipeline import RAGPipeline

# Initialize the RAGPipeline
pipeline = RAGPipeline()

# Set Streamlit page configuration
st.set_page_config(page_title="Research Paper Query System", page_icon="📚", layout="wide")

# Title of the app
st.title("📚 Research Paper Query System")

# Sidebar for file selection (Accessing files from existing folder)
with st.sidebar:
    st.header("Access Existing PDFs")
    folder_path = "/Users/sravyasarath/Documents/Projects/Practice/GenAI/DMS/input"  # Path to your existing folder
    if not os.path.exists(folder_path):
        st.error(f"The folder '{folder_path}' does not exist!")
    else:
        files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        if files:
            selected_file = st.selectbox("Select a PDF file", files)
            if selected_file:
                st.write(f"**Selected file**: {selected_file}")
                # Process the selected file
                processed_papers = pipeline.process_pdfs([selected_file])
                st.success(f"File '{selected_file}' processed successfully!")
        else:
            st.write(f"No PDFs found in the folder '{folder_path}'.")

# Display available documents after processing
if pipeline.papers_info:
    st.header("Available Research Papers")
    paper_list = pipeline.list_papers()
    st.text(paper_list)

    # User selects a document to query
    selected_paper = None
    selected_paper = pipeline.select_document(0)  # Automatically select the first document for simplicity
    st.write(selected_paper)

    # Start conversation
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Query section - Add text input to ask questions
    user_query = st.text_area("Enter your query:")

    if user_query:
        # Get AI response
        response = pipeline.query_documents(user_query)

        # Append the conversation to session state
        st.session_state.conversation_history.append({"user": user_query, "ai": response})

        # Display the conversation in the chatbot style
        for chat in st.session_state.conversation_history:
            st.markdown(f'<div style="text-align:left;"><b>AI:</b><p>{chat["ai"]}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:right;"><b>You:</b><p>{chat["user"]}</p></div>', unsafe_allow_html=True)

    # Button to close the conversation
    close_button = st.button("Close Conversation")
    if close_button:
        # Reset the conversation history
        st.session_state.conversation_history = []
        st.write("Conversation closed. Start a new query!")

else:
    st.write("Please access and process PDFs from the existing folder.")
