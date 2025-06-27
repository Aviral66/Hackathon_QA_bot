# AskIt - Simple Document Q&A App

## 🧩 Problem Statement

**Objective:** Build a lightweight internal assistant that can accurately answer employee queries by referencing company documents, policies, and procedures.

**Example Queries:**

- What is our notice period policy?
- Can I encash unused leave?
- Who do I contact for laptop issues?

**AskIt** is a streamlined, powerful tool designed to help employees interact naturally with internal documents using state-of-the-art Large Language Models (LLMs) and vector similarity search. By supporting multiple file formats (PDFs, DOCX, TXT, and web links), AskIt empowers teams to quickly access relevant policies, extract insights, and get reliable answers without digging through manual files or contacting HR for every question.

## ✅ Features

- **Multiple File Format Support**: Analyze text from PDFs, DOCX, TXT files, or even web URLs.
- **Flexible LLM and Embedding Choices**: Switch between various local or cloud-based models for both embedding and answering questions.
- **Streamlined UI**: Built with Streamlit for a clean and responsive interface.
- **Privacy-First Approach**: Everything runs locally; your data never leaves your machine.
- **Real-Time Querying**: Ask questions about documents and get answers instantly.
- **Chunking and Indexing**: Large documents are split into manageable text segments for better semantic matching.
- **Multi-Document & Multi-Link Support**: Load and query across several files or websites at once.
- **Visual and Text-Based Outputs**: Optionally extract images or tables (PDFs) in future versions.

---

## 🚀 Quick Start

### 1. Install Dependencies

First, set up the Python environment and install required libraries:

```bash
pip install -r requirements.txt
```

Your environment should include (but is not limited to):

- `faiss-cpu` – for vector similarity search
- `streamlit` – for the front-end interface
- `langchain` – to integrate LLM pipelines
- `PyPDF2`, `pymupdf` – for PDF reading
- `python-docx` – for DOCX file parsing
- `huggingface_hub` – for model access
- `ollama` – for running local LLMs

---

### 2. Set Up Environment Variables

Create a `.env` file to store optional settings or API keys:

```bash
touch .env
# Add your keys or settings in KEY=VALUE format
```

Examples:

```
HUGGINGFACE_API_KEY=your_key_here
MODEL_PROVIDER=ollama
```

---

### 3. Run the App

Once setup is complete, launch the Streamlit app:

```bash
streamlit run app.py
```

Ensure Ollama is running locally if you plan to use local LLMs.

---

## ⚙️ Configuration

### Embedding Models

These models convert text into high-dimensional vectors for similarity matching:

- `all-mpnet-base-v2` – High-performance general-purpose embeddings
- `all-MiniLM-L6-v2` – Fast and memory-efficient
- `paraphrase-multilingual-MiniLM-L12-v2` – Great for multilingual queries

### LLM Models (via Ollama)

Used for final answer generation:

- `gemma3:4b` – Lightweight and effective general use model
- `deepseek-r1:latest` – Ideal for factual QA and summarization
- `llama3.2:latest` – Meta’s LLM optimized for deep contextual understanding

> You can add more models to Ollama's configuration based on system capabilities.

---

## 🧪 Example Usage

**Sample Workflow:**

1. Drop your source files (e.g., policy documents) into the `docs/` directory.
2. Ingest the content using one of two methods:
   - **Path A**: Upload via Streamlit interface.
   - **Path B**: Run a script or CLI command to load documents into the vector store.
3. Launch the chat interface by running the Streamlit app.
4. Ask natural-language questions like:
   - What are the key policies outlined?
   - Who are the stakeholders mentioned?
   - Summarize section 3 in simple terms.
   - List all numerical data related to annual sales.
5. Receive AI-generated responses, each with cited context snippets for transparency.

AskIt matches your query to relevant parts of the content, then uses an LLM to generate a contextual response.

---

## 📁 Project Structure

```
askit/
├── app.py               # Main Streamlit app logic
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (optional)
└── README.md            # Project documentation (this file)
```

---

## 🧠 How It Works

1. **File/Web Input**: Accepts documents or web URLs for analysis.
2. **Text Parsing & Splitting**: Extracts and chunks text intelligently.
3. **Embedding**: Converts chunks to vectors using an embedding model.
4. **FAISS Indexing**: Stores vectors for fast semantic search.
5. **Query Matching**: Finds most relevant chunks to your question.
6. **Answer Generation**: Feeds the result to an LLM to construct a response.

This modular design ensures both speed and relevance in the response process.

---

## 📜 License

MIT License © [Your Name]

Free to use, modify, and distribute under the terms of the MIT license.

---

## 🤝 Contribute

We welcome contributions of all kinds! Whether it’s fixing a bug, improving documentation, or adding new features:

- Fork the repo
- Create a new branch
- Submit a pull request with clear changes

Let’s build smarter document tools together!
