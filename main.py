import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

st.title("🧠 Fireflies Market Intelligence Assistant")

# Step 1: Upload Transcript
uploaded_file = st.file_uploader("Upload a Fireflies Meeting Transcript (.txt)", type="txt")

if uploaded_file:
    with open("temp_transcript.txt", "wb") as f:
        f.write(uploaded_file.read())

    # Step 2: Load and Filter Transcript
    loader = TextLoader("temp_transcript.txt")
    docs = loader.load()

    # Step 3: Split text into chunks and filter for market-related content
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = splitter.split_documents(docs)

    market_keywords = ["market", "tam", "sam", "trend", "growth", "opportunity", "customer", "size"]
    market_chunks = [chunk for chunk in all_chunks if any(kw.lower() in chunk.page_content.lower() for kw in market_keywords)]

    # Step 4: Embed and Store in Chroma
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(market_chunks, embedding)

    # Step 5: Build and Run QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=vectordb.as_retriever()
    )

    query = st.text_input("Ask a market-related question from the meeting:")
    if query:
        response = qa_chain.run(query)
        st.write("💬", response)
