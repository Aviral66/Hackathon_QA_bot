import faiss
import streamlit as st
import os
import numpy as np
from io import BytesIO
from docx import Document
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader, PDFPlumberLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import fitz  # PyMuPDF
import tempfile

# Load environment variables
load_dotenv()

# Initialize the Streamlit app
st.set_page_config(page_title="AskIt - Q&A with Multiple Files", layout="wide")

# Sidebar for model selection
st.sidebar.header("Model Selection")
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ]
)

llm_model = st.sidebar.selectbox(
    "Select LLM Model",
    [
        "gemma3:4b",
        "deepseek-r1:latest",
        "llama3.2:latest"
    ]
)

file_type = st.sidebar.selectbox(
    "Select File Type",
    ["Link", "PDF", "Text", "DOCX", "TXT"]
)

# Initialize the selected LLM model
ollama_model = OllamaLLM(model=llm_model)


def load_pdf(file):
    """Load PDF documents using PyMuPDF and extract images."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    doc = fitz.open(temp_file_path)
    text = ""
    images = []
    for page in doc:
        text += page.get_text()  # Extract text from the page
        for img_index in range(len(page.get_images(full=True))):
            img = page.get_images(full=True)[img_index]
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)  # Store image bytes
    doc.close()
    os.remove(temp_file_path)
    return text, images


def split_text(documents):
    """Split documents into chunks using CharacterTextSplitter."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)


def process_input(input_type, input_data):
    """Processes different input types and returns a vectorstore."""
    documents = ""
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        web_documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = [str(doc.page_content) for doc in text_splitter.split_documents(web_documents)]
    elif input_type == "PDF":
        for file in input_data:
            pdf_reader = PdfReader(BytesIO(file.read()))
            for page in pdf_reader.pages:
                documents += page.extract_text()
    elif input_type == "Text":
        documents = input_data
    elif input_type == "DOCX":
        for file in input_data:
            doc = Document(BytesIO(file.read()))
            documents += "\n".join([para.text for para in doc.paragraphs])
    elif input_type == "TXT":
        for file in input_data:
            documents += file.read().decode('utf-8')
    else:
        raise ValueError("Unsupported input type")

    if input_type != "Link":
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(documents)
    else:
        texts = documents  # For links, we already have the texts

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    # Create FAISS index
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)

    # Create FAISS vector store with the embedding function
    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Add documents and their embeddings to the vector store
    vector_store.add_texts(texts)  # Add documents to the vector store
    for i, text in enumerate(texts):
        vector_store.index.add(np.array([hf_embeddings.embed_query(text)]).astype(np.float32))  # Add embeddings

    return vector_store


def answer_question(vectorstore, query):
    try:
        # Prompt engineering for HR policy context
        modified_query = f"You are an HR assistant. Based on the following HR policy documents, answer the question concisely: {query}"

        qa = RetrievalQA.from_chain_type(
            llm=ollama_model,
            chain_type="stuff",  # You can also try "map_reduce" for different behavior
            retriever=vectorstore.as_retriever()
        )

        # Debugging: Check if the vectorstore is empty
        if not vectorstore:
            return "The vector store is empty. Please process the input first."

        result = qa({"query": modified_query})

        # Debugging: Check the result structure
        if not result or "result" not in result:
            return "No valid result returned from the model."

        answer = result.get("result", "No answer found.")
        return answer

    except Exception as e:
        return f"An error occurred: {str(e)}"


def main():
    st.title("AskIt - Q&A with Multiple Files")
    st.write("Upload documents or provide a link to ask questions based on their content.")

    input_data = None

    if file_type == "Link":
        number_of_links = st.number_input("Number of Links", min_value=1, max_value=20, step=1)
        input_data = [st.text_input(f"Link {i + 1}") for i in range(int(number_of_links))]
        input_data = [link for link in input_data if link.strip()]  # Remove empty links
    elif file_type == "Text":
        input_data = st.text_area("Enter the text")
    elif file_type in ["PDF", "DOCX", "TXT"]:
        input_data = st.file_uploader(
            f"Upload {file_type} files", type=[file_type.lower()], accept_multiple_files=True
        )

    if st.button("Process Input"):
        if file_type == "Link" and not input_data:
            st.error("Please provide at least one valid link.")
        elif file_type != "Link" and not input_data:
            st.error("Please upload at least one file or enter valid text.")
        else:
            try:
                vectorstore = process_input(file_type, input_data)
                st.session_state["vectorstore"] = vectorstore
                st.session_state["chat_active"] = True  # Set chat active state
                st.success("Input processed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing the input: {str(e)}")

    if "vectorstore" in st.session_state and st.session_state.get("chat_active", False):
        st.subheader("Chat with the Bot")
        query = st.text_input("Ask your question:")
        if st.button("Submit"):
            if query.strip():
                answer = answer_question(st.session_state["vectorstore"], query)
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].append((query, answer))
                # No need to call st.experimental_rerun()
            else:
                st.error("Please enter a valid question.")

        # Display chat history
        if "history" in st.session_state and st.session_state["history"]:
            for q, a in st.session_state["history"]:
                st.markdown(f"<div style='text-align: right;'><b>You:</b> {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: left;'><b>Bot:</b> {a}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
