import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="Bhagavad Gita GPT", page_icon="🕉️")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.image("images/krishna.png", width=300)  
st.sidebar.markdown("""
# Welcome to the Bhagavad Gita Q&A

The **Bhagavad Gita** is a 700-verse Hindu scripture, written as a dialogue 
between Prince Arjuna and Lord Krishna. Its teachings on duty, righteousness, 
and spirituality continue to inspire millions worldwide.

Feel free to ask questions and explore its timeless wisdom!
""")

# ── Main header ───────────────────────────────────────────
col1, col2 = st.columns([1, 7])
with col1:
    st.image("images/sacred_book_1.png", width=100)
with col2:
    st.markdown("<h1>Bhagavad Gita Q&A</h1>", unsafe_allow_html=True)

# ── Backend ───────────────────────────────────────────────
VECTOR_STORE_PATH = "vector_store/index.faiss"

def initialize_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  

    if os.path.exists(VECTOR_STORE_PATH):
        # Load existing FAISS index
        return FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        # Create new FAISS index
        loader = PyPDFDirectoryLoader("./pdfs")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500
        )
        chunks = splitter.split_documents(docs)

        vectors = FAISS.from_documents(chunks, embeddings)
        vectors.save_local(VECTOR_STORE_PATH)
        return vectors

def build_chain(vectors):
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile", 
        temperature=0.2
    )

    retriever = vectors.as_retriever(search_kwargs={"k": 3})

    # History aware retriever
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and latest question, reformulate it as a standalone question. Don't answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Answer prompt ✅ updated from old HumanMessagePromptTemplate style
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a highly knowledgeable expert in the Bhagavad Gita.
         Answer ONLY from the context below. If the answer is not in the context,
         say 'I don't know.' Reference specific verses where applicable.
         
         Context: {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    document_chain  = create_stuff_documents_chain(llm, answer_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    return retrieval_chain

# ── Session state ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner("Loading Bhagavad Gita knowledge base..."):
        vectors = initialize_vector_store()
        st.session_state.chain = build_chain(vectors)

# ── Chat UI ───────────────────────────────────────────────
st.markdown("### 🙏 Ask your questions to Lord Krishna!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Be precise with your question for a better response..."):

    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Seeking wisdom..."):
            result = st.session_state.chain.invoke({
                "input": question,
                "chat_history": st.session_state.chat_history
            })
            answer = result["answer"]
            st.write(answer)

            with st.expander("📖 Source Verses"):
                for i, doc in enumerate(result["context"]):
                    st.markdown(f"**Source {i+1} (page {doc.metadata.get('page', '?')}):**")
                    st.caption(doc.page_content[:300])

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
