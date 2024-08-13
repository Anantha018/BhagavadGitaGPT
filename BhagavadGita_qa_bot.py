import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
import hashlib
import pickle

# Load environment variables from a .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")  # Retrieve the API key for Groq

# Streamlit setup
st.set_page_config(page_title="Bhagavad Gita GPT", page_icon="🕉️")  # Set the page configuration for the Streamlit app

# Display an image of Krishna in the sidebar
st.sidebar.image("images/krishna.png", use_column_width=True)

# Display information about the Bhagavad Gita in the sidebar
st.sidebar.markdown("""
# Welcome to the Bhagavad Gita Q&A

## What is the Bhagavad Gita?

The **Bhagavad Gita**, often referred to simply as the **Gita**, is a 700-verse Hindu scripture that is part of the Indian epic Mahabharata. It is a sacred text of the Hindu religion and is considered one of the most important spiritual classics in history. The Gita is written in the form of a dialogue between Prince Arjuna and the god Krishna, who serves as his charioteer.

## Why is the Bhagavad Gita Important?

### Spiritual and Philosophical Significance
The Bhagavad Gita addresses the moral and philosophical dilemmas faced by Arjuna on the battlefield of Kurukshetra. It provides profound insights into duty, righteousness, and the nature of reality. The Gita discusses various paths to spiritual enlightenment, including devotion, knowledge, and disciplined action. Its teachings have influenced countless individuals and continue to offer guidance on leading a life of virtue and wisdom.

### Universal Teachings
The Gita transcends religious boundaries and speaks to universal human concerns. Its teachings on the nature of the self, the importance of duty, and the pursuit of spiritual knowledge resonate with people from various cultural and spiritual backgrounds. The text emphasizes the importance of living according to one's principles while remaining detached from the outcomes of one's actions.

### Historical and Cultural Impact
The Bhagavad Gita has had a significant impact on Indian culture and philosophy, and its influence extends globally. It has been studied and commented upon by numerous scholars, philosophers, and spiritual leaders. The Gita's teachings have inspired various movements and continue to be a source of inspiration for personal and collective transformation.

Feel free to ask questions about the Bhagavad Gita, and explore its timeless wisdom through this application!
""")

# Path to your image file
image_path = "images/sacred_book_1.png"  # Replace with the path to your image file

# Display the main image and title on the main page
col1, col2 = st.columns([1, 7])
with col1:
    st.image(image_path, width=100)  # Display the image with a width of 100 pixels
with col2:
    st.markdown("<h1 style='display: flex; align-items: center;'>Bhagavad Gita Q&A</h1>", unsafe_allow_html=True)

# Set up the language model for answering questions
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Define the prompt template for generating responses
template_string = """
You are a highly knowledgeable expert in the Bhagavad Gita, familiar with its teachings, context, and interpretations. Your task is to provide a well-informed, precise, and contextually accurate answer based on the provided excerpt from the Bhagavad Gita.

**Context:**
{context}

**Question:**
{input}

**Instructions:**
- Analyze the provided context thoroughly to ensure your answer is grounded in the specific teachings or verses from the Bhagavad Gita.
- Your response should directly address the question posed, reflecting the philosophical, spiritual, or practical insights found in the Bhagavad Gita.
- If applicable, reference specific verses or concepts from the text to support your answer.
- Ensure clarity and coherence in your explanation, avoiding unnecessary jargon and making complex ideas accessible.

**Answer:**
"""

# Create a ChatPromptTemplate instance from the prompt template string
prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(template_string)
])

# Define paths for storing vector and metadata files
VECTOR_STORE_PATH = "vector_store/faiss_index"
METADATA_STORE_PATH = "vector_store/doc_metadata.pkl"

def compute_doc_hash(doc):
    """Compute a hash for a document to identify changes."""
    return hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()

# Dont call this function once the db is created, so instead of commenting this just remove or keep ore move the folder files of gita folder somewhere
 
def initialize_vector_store():
    """Initialize the vector store: create it if it doesn't exist, or load it if it does, and update with new documents."""
    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists(VECTOR_STORE_PATH):
        # Load existing vector store if it exists
        vectors = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
        if os.path.exists(METADATA_STORE_PATH):
            # Load existing metadata if it exists
            with open(METADATA_STORE_PATH, 'rb') as f:
                existing_metadata = pickle.load(f)
        else:
            existing_metadata = {}
    else:
        # Create a new vector store if it does not exist
        vectors = None
        existing_metadata = {}

    # Load and split documents
    loader = PyPDFDirectoryLoader("./BhagavadGita")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    new_documents = text_splitter.split_documents(docs)
    
    new_embeddings = []
    new_doc_hashes = []
    
    for doc in new_documents:
        doc_hash = compute_doc_hash(doc)
        if doc_hash not in existing_metadata:
            # If the document is new or has changed, embed and add it to the index
            new_embeddings.append(doc)
            new_doc_hashes.append(doc_hash)
    
    if new_embeddings:  # If there are new documents
        if vectors is None:
            vectors = FAISS.from_documents(new_embeddings, embeddings)
        else:
            vectors.add_documents(new_embeddings)
        
        # Update the metadata store with new documents
        for doc, doc_hash in zip(new_embeddings, new_doc_hashes):
            existing_metadata[doc_hash] = doc.metadata
        
        # Save the updated vector store and metadata
        vectors.save_local(VECTOR_STORE_PATH)
        with open(METADATA_STORE_PATH, 'wb') as f:
            pickle.dump(existing_metadata, f)
    else:
        st.write("")
        # st.write("No new documents to embed.")
    
    return vectors

# Initialize the vector store before rendering the UI
vectors = initialize_vector_store()

# Add custom HTML and CSS to style the label
st.markdown("""
    <style>
    .custom-label {
        font-size: 24px; /* Adjust the font size */
        font-weight: bold; /* Make the text bold */
        margin-bottom: 10px; /* Add some space below the label */
        align-items: center; /* Center the label horizontally */
        display: flex;
    }
    </style>
    <div class="custom-label">
        Ask your questions to Lord Krishna!
    </div>
""", unsafe_allow_html=True)

# Text input field with an empty label
prompt1 = st.text_input(
    label=" ",  # Empty label
    placeholder="Be precise with your question for a better response..."
)

if prompt1:
    # Set up retrieval QA chain for answering questions
    retriever = vectors.as_retriever()
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Get the response from the QA chain
    response = retrieval_qa_chain({"query": prompt1})
    st.write(response["result"])
