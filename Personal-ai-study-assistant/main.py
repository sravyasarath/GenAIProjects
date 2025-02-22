from rag_pipeline import RAGPipeline

def main():
    """Handles user interaction for document selection & query processing."""
    rag = RAGPipeline()
    rag.process_pdfs()  # Load and process PDFs

    print("\n👋 **Welcome to AI Research Assistant!**")
    print(rag.list_papers())

    while True:
        user_input = input("\nSelect a document (number) or ask a query: ").strip()

        # Handle document selection
        if user_input.isdigit():
            doc_index = int(user_input) - 1
            print(rag.select_document(doc_index))
            continue

        # Exit condition
        if user_input.lower() == "exit":
            print("👋 Goodbye! Have a great day!")
            break

        # Handle query after document selection
        if not rag.selected_doc:
            print("❌ Please select a document before asking queries.")
            continue

        # Retrieve and answer query using RAG + ChatGroq
        answer = rag.query_documents(user_input)
        print(f"\n📝 **AI Response:**\n{answer}")

if __name__ == "__main__":
    main()
