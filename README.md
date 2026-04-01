# 🕉️ Bhagavad Gita GPT

A conversational AI that answers questions about the Bhagavad Gita using Retrieval-Augmented Generation (RAG). Ask anything about the Gita's teachings, verses, and philosophy — and get accurate, context-grounded answers.

---

## ✨ Features

- 📖 **PDF-based knowledge** — answers grounded strictly in the Bhagavad Gita text
- 🧠 **Chat memory** — remembers previous questions within a session
- 🔍 **Source transparency** — shows which verses/pages the answer came from
- 🚫 **Hallucination guard** — only answers from the document, never from general knowledge
- ⚡ **Fast responses** — powered by Groq's ultra-fast LLM inference
- 💾 **Persistent vector store** — FAISS index built once, reused every run

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit |
| **LLM** | Groq (`llama-3.3-70b-versatile`) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS (local) |
| **RAG Framework** | LangChain |
| **PDF Loader** | PyPDFDirectoryLoader |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/BhagavadGitaGPT.git
cd BhagavadGitaGPT
```

### 2. Create and activate virtual environment
```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# Mac/Linux
source myenv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at → https://console.groq.com

### 5. Add your PDFs

Place your Bhagavad Gita PDF(s) inside the `pdfs/` folder:
```
BhagavadGitaGPT/
└── pdfs/
    └── bhagavad_gita.pdf
```

---

## 🚀 How to Use

### First run — builds the vector store automatically:
```bash
streamlit run BhagavadGita.py
```

The app will:
1. Load PDFs from the `pdfs/` folder
2. Chunk and embed the text
3. Save the FAISS index to `vector_store/`
4. Launch the chat interface at `http://localhost:8501`

### Subsequent runs — loads from disk instantly:
```bash
streamlit run BhagavadGita.py
```
No re-ingestion needed — the vector store is reused automatically.

---

## 📁 Project Structure

```
BhagavadGitaGPT/
│
├── pdfs/                   # Place your Bhagavad Gita PDFs here
├── images/                 # UI images (krishna.png, sacred_book_1.png)
├── vector_store/           # Auto-generated FAISS index (gitignored)
├── BhagavadGita.py         # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .env                    # API keys (gitignored)
└── .gitignore
```

---

## ⚙️ Requirements

```txt
streamlit
langchain==0.3.15
langchain-community==0.3.15
langchain-groq
langchain-huggingface
langchain-text-splitters
langchain-core
sentence-transformers
faiss-cpu
pypdf
python-dotenv
```



## 🙏 Acknowledgements

- [Bhagavad Gita](https://en.wikipedia.org/wiki/Bhagavad_Gita) — the sacred text
- [LangChain](https://langchain.com) — RAG framework
- [Groq](https://groq.com) — LLM inference
- [Streamlit](https://streamlit.io) — UI framework