from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from gen_ai_hub.proxy.langchain import OpenAIEmbeddings
from langchain_community.vectorstores.hanavector import HanaDB
from langchain.docstore.document import Document
from typing import List
from  hdbcli import dbapi
from gen_ai_hub.proxy.native.openai import embeddings
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
import streamlit as st
from gen_ai_hub.proxy.langchain.init_models import init_llm
from gen_ai_hub.proxy.langchain import OpenAIEmbeddings
from langchain_community.vectorstores.hanavector import HanaDB
from langchain.docstore.document import Document
from typing import List
from hdbcli import dbapi
import hana_ml.dataframe as dataframe
import os
from dotenv import load_dotenv
from gen_ai_hub.proxy.native.openai import 

#Use this function to load the documents from your folder where you downloaded it into, we will use this later to split and chunk the documents.
def load_documents():
    document_loader = PyPDFDirectoryLoader("C:/Users/I757543/Downloads/datafolder")
    return document_loader.load()


def split_documents(documents: list[Document]):
    # Create a text splitter that breaks documents into chunks of up to 800 characters
    # with an overlap of 80 characters between consecutive chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,        # Length is measured by character count
        is_separator_regex=False,   # Separators are treated as plain strings, not regex
    )
    # Split the input documents into smaller chunks
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks):
    # Tracks the page ID of the last processed chunk
    last_page_id = None
    # Tracks the index of the chunk within the current page
    current_chunk_index = 0

    for chunk in chunks:
        # Extract source and page metadata from the chunk
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"  # Unique identifier for the page

        # If the chunk belongs to the same page as the previous one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            # If the page has changed, reset the chunk index
            current_chunk_index = 0

        # Create a unique chunk ID in the format: source:page:chunk_index
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        # Update last_page_id for the next iteration
        last_page_id = current_page_id

        # Store the generated chunk ID in the chunk's metadata
        chunk.metadata["id"] = chunk_id

    return chunks

# Load environment variables
load_dotenv()
HANA_USER_VDB = os.getenv('HANA_VECTOR_USER')
HANA_PASSWORD_VDB = os.getenv('HANA_VECTOR_PASS')
HANA_HOST = os.getenv('HANA_HOST_VECTOR')
SCHEMA_NAME = "***"
TABLE_NAME = "***"

# Connect to HANA
conn = dataframe.ConnectionContext(

)

conn1 = dbapi.connect(
    
)

if not conn.has_table(table=TABLE_NAME, schema=SCHEMA_NAME):
    conn.create_table(table=TABLE_NAME, schema=SCHEMA_NAME, table_structure={'FILENAME':'NVARCHAR(100)','TEXT':'NCLOB','VECTOR':'REAL_VECTOR(1536)'})


def save_to_hana(chunks, conn):
    cursor = conn.cursor()
    insert_sql = f"INSERT INTO {SCHEMA_NAME}.{TABLE_NAME} (FILENAME, TEXT, VECTOR) VALUES (?, ?, ?)"
    
    for chunk in chunks:
        filename = chunk.metadata.get('source')
        text = chunk.page_content
        vector = generate_embeddings(text)  # returns List[float]

        cursor.execute(insert_sql, (filename, text, vector))  # vector passed as list
    
    conn.commit()
    cursor.close()

documents = load_documents()
chunks = split_documents(documents)

# Function to generate embeddings
def generate_embeddings(text, model="text-embedding-ada-002"):
    return embeddings.create(input=[text], model=model).data[0].embedding


def semantic_search(prompt, conn, embeddings_client, top_k=10):
    embedding = embeddings_client.create(input=[prompt], model="text-embedding-ada-002")
    query_vector = embedding.data[0].embedding
    query_vector_str = str(query_vector)
    
    cursor = conn.cursor()
    sql = '''
        SELECT "TEXT"
        FROM "AWSEMBEDDING"."AWSTARGET"
        ORDER BY COSINE_SIMILARITY("VECTOR", TO_REAL_VECTOR(?)) DESC
        LIMIT ?
    '''
    cursor.execute(sql, (query_vector_str, top_k))
    results = cursor.fetchall()
    cursor.close()
    return [r[0] for r in results]

st.title("Lambda Documentation Search")

prompt = st.text_area("Enter your prompt:")

if st.button("Search & Summarize"):
    if prompt:
        with st.spinner("Fetching relevant documents..."):
            chunks = semantic_search(prompt, conn1, embeddings, top_k=10)
            context = "\n\n".join(chunks)

        llmPrompt = f"""
        You are an AI assistant specialized in document analysis and summarization. 
        Analyze the given documentation and summarize it. Provide page numbers when possible. DO NOT USE ANY DATA YOU HAVE ALREADY. USE ONLY WHAT I GIVE YOU.
        PROVIDE THE PAGE NUMBERS AND WHERE YOU GOT THE DATA FROM AS REFERENCED BY MY CONTEXT

        Context (top 10 documents): 
        {context}

        
        """

        llm = init_llm('gpt-4o-mini', temperature=0., max_tokens=512)
        response = llm.invoke(llmPrompt).content
        st.subheader("Response")
        st.write(response)
