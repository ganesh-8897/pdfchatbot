import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
import os
import time

# App Configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="centered"
)

# Initialize components
@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model="llama3")

@st.cache_resource
def load_llm():
    return Ollama(model="llama3", temperature=0.3)

# PDF Processing Functions
def extract_text(pdf_file):
    """Extract text from PDF with error handling"""
    try:
        reader = PdfReader(pdf_file)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"Failed to extract text: {str(e)}")
        return None

def process_pdf(pdf_file, embeddings):
    """Process PDF and create vector store"""
    with st.spinner("Processing PDF..."):
        start_time = time.time()
        
        # Extract and split text
        raw_text = extract_text(pdf_file)
        if not raw_text:
            return None
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(raw_text)
        
        # Create/update vector store
        if os.path.exists("faiss_index"):
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            db.add_texts(chunks)
        else:
            db = FAISS.from_texts(chunks, embeddings)
        
        db.save_local("faiss_index")
        st.success(f"Processed {len(chunks)} chunks in {time.time()-start_time:.1f}s")
        return db

# UI Components
def sidebar():
    with st.sidebar:
        st.title("Settings")
        if st.button("Clear Database"):
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
                st.success("Database cleared!")

def main():
    st.title("PDF Chat Assistant")
    st.caption("Upload a PDF and ask questions about its content")
    
    # File upload
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if pdf_file:
        embeddings = load_embeddings()
        llm = load_llm()
        
        # Process PDF
        db = process_pdf(pdf_file, embeddings)
        if not db:
            return
            
        # Question answering
        question = st.text_input("Ask about the PDF:")
        if question:
            with st.spinner("Finding answer..."):
                try:
                    docs = db.similarity_search(question, k=3)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    answer = chain.run(input_documents=docs, question=question)
                    st.markdown("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    sidebar()
    main()