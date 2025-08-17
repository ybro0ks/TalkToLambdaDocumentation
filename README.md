Lambda Documentation Search

A Streamlit-based application that allows users to perform semantic search and summarization over a collection of PDF documents using embeddings stored in SAP HANA.

Features

PDF Document Loading: Automatically loads PDFs from a local folder.

Text Chunking: Splits documents into manageable chunks with overlap for improved search accuracy.

Embeddings Storage: Stores text embeddings in SAP HANA for fast semantic search.

Semantic Search: Retrieves top-k most relevant document chunks based on user queries using cosine similarity.

AI Summarization: Summarizes relevant documents with page references using GPT-4o-mini.

Installation

Clone the repository:

git clone <repo-url>
cd <repo-folder>


Install dependencies:

pip install -r requirements.txt


Set up environment variables in a .env file:

HANA_VECTOR_USER=<your_hana_user>
HANA_VECTOR_PASS=<your_hana_password>
HANA_HOST_VECTOR=<hana_host>

Usage

Place your PDFs in a folder, e.g., datafolder.

Run the Streamlit app:

streamlit run app.py


Enter a prompt in the text area and click Search & Summarize.

View the summarized results with references to the source documents.

Functions Overview

load_documents(): Loads all PDFs from a local folder.

split_documents(documents): Splits text into overlapping chunks.

calculate_chunk_ids(chunks): Assigns unique IDs to each text chunk.

generate_embeddings(text): Converts text chunks into vector embeddings using OpenAI.

save_to_hana(chunks, conn): Stores text chunks and embeddings in SAP HANA.

semantic_search(prompt, conn, embeddings_client, top_k): Retrieves the most relevant chunks for a query.

Requirements

Python 3.9+

Streamlit

SAP HANA Client (hdbcli)

langchain and related modules

OpenAI API access

License

MIT License
