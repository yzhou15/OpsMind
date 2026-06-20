#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpsMind Intelligent Operations Assistant - Streamlit Frontend"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_BASE_URL"] = "https://hf-mirror.com"

import streamlit as st
import time

from configs.settings import EmbeddingConfig, ChunkConfig

st.set_page_config(
    page_title="OpsMind Intelligent Operations Assistant",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource(show_spinner=False)
def initialize_services():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
    from src.rag.vector_store import VectorStoreManager

    embeddings = HuggingFaceEmbeddings(model_name=EmbeddingConfig.MODEL_NAME)
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)
    
    store = VectorStoreManager(
        persist_dir="./chroma_db",
        embedding_model=EmbeddingConfig.MODEL_NAME
    )
    db = store._get_db()
    
    return embeddings, llm, db, store


with st.spinner("Initializing services..."):
    embeddings, llm, db, store = initialize_services()

st.title("🤖 OpsMind Intelligent Operations Assistant")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Select Model", ["qwen2.5:7b", "llama3:7b"], index=0)
    top_k = st.slider("Retrieval Count", min_value=1, max_value=5, value=3)


def get_rag_response(question, top_k=3):
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)
    
    if not docs:
        return None, []
    
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    from langchain_core.prompts import ChatPromptTemplate
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a senior operations expert. Answer questions based on the provided knowledge base. If the knowledge base doesn't have relevant information, clearly state this."),
        ("human", "Knowledge Base:\n{context}\n\nQuestion:\n{question}")
    ])
    
    chain = template | llm
    response = chain.invoke({"context": context, "question": question})
    
    return response.content, docs


tab1, tab2, tab3 = st.tabs(["Q&A", "Data Management", "System Status"])

with tab1:
    st.subheader("Knowledge Q&A")
    
    question = st.text_input("Ask a question about operations:", key="question_input")
    
    if question:
        with st.spinner("Searching knowledge base..."):
            start_time = time.time()
            
            response_content, docs = get_rag_response(question, top_k)
            
            elapsed_time = time.time() - start_time
            
            if response_content:
                st.markdown("### 📝 Answer")
                st.write(response_content)
                st.caption(f"Response time: {elapsed_time:.2f} seconds")
                
                st.markdown("### 📚 Reference Sources")
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"Source {i}"):
                        st.write(doc.page_content)
                        if hasattr(doc, 'metadata'):
                            st.caption(f"Tags: {doc.metadata.get('tags', '')} | Source: {doc.metadata.get('source_file', '')}")
            else:
                st.warning("No relevant information found in knowledge base")

with tab2:
    st.subheader("Data Management")
    
    doc_count = store.get_document_count()
    st.info(f"Current document count: {doc_count}")
    
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "json", "xml", "log", "pdf", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ChunkConfig.CHUNK_SIZE,
            chunk_overlap=ChunkConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        for uploaded_file in uploaded_files:
            file_path = f"data/raw/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            from langchain_community.document_loaders import TextLoader, PyPDFLoader
            
            try:
                if file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding="utf-8")
                
                docs = loader.load()
                split_docs = text_splitter.split_documents(docs)
                
                for doc in split_docs:
                    doc.metadata["source_file"] = uploaded_file.name
                
                store.add_documents([d.page_content for d in split_docs], 
                                   [d.metadata for d in split_docs])
                
                st.success(f"✅ Added: {uploaded_file.name} ({len(split_docs)} chunks)")
            except Exception as e:
                st.error(f"❌ Failed to add {uploaded_file.name}: {e}")
    
    if st.button("Clear Database"):
        store.clear_all()
        st.success("✅ Database cleared")

with tab3:
    st.subheader("System Status")
    
    st.info(f"Embedding Model: {EmbeddingConfig.MODEL_NAME}")
    st.info(f"Chunk Size: {ChunkConfig.CHUNK_SIZE}, Overlap: {ChunkConfig.CHUNK_OVERLAP}")
    
    try:
        response = llm.invoke("test")
        st.success("✅ Ollama connection OK")
    except Exception as e:
        st.error(f"❌ Ollama connection failed: {e}")
    
    try:
        count = store.get_document_count()
        st.success(f"✅ Vector DB OK, document count: {count}")
    except Exception as e:
        st.error(f"❌ Vector DB error: {e}")

st.markdown("---")
st.markdown("OpsMind Intelligent Operations Assistant | Powered by RAG & Large Language Models")
