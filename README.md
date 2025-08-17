# Lambda Documentation Search

A Streamlit-based application that allows users to perform semantic search and summarization over a collection of PDF documents using embeddings stored in SAP HANA.

## Features

- **PDF Document Loading**: Automatically loads PDFs from a local folder.
- **Text Chunking**: Splits documents into manageable chunks with overlap for improved search accuracy.
- **Embeddings Storage**: Stores text embeddings in SAP HANA for fast semantic search.
- **Semantic Search**: Retrieves top-k most relevant document chunks based on user queries using cosine similarity.
- **AI Summarization**: Summarizes relevant documents with page references using GPT-4o-mini.

Functions Overview

load_documents(): Loads all PDFs from a local folder.

split_documents(documents): Splits text into overlapping chunks.

calculate_chunk_ids(chunks): Assigns unique IDs to each text chunk.

generate_embeddings(text): Converts text chunks into vector embeddings using OpenAI.

save_to_hana(chunks, conn): Stores text chunks and embeddings in SAP HANA.

semantic_search(prompt, conn, embeddings_client, top_k): Retrieves the most relevant chunks for a query.
