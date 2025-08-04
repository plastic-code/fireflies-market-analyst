import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="🦜🔍 Fireflies Market Assistant")
st.title("🦜🔍 Fireflies Market Assistant")
st.markdown("Upload a Fireflies meeting transcript and get market intelligence insights using GPT.")

uploaded_file = st.file_uploader("Upload a Fireflies transcript (.txt)", type="txt")
question = st.text_input("Ask a specific question (optional):")

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        # Load and split
        loader = TextLoader(tmp_file_path)
        docs = loader.load()

        # Split into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(docs)

        # Setup OpenAI embedding model
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

        # Embed and store in FAISS
        db = FAISS.from_documents(texts, embeddings)

        retriever = db.as_retriever()
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"]),
            chain_type="stuff",
            retriever=retriever
        )

        # Run query
        query = question if question else "What are the main market insights from this transcript?"
        with st.spinner("Analyzing..."):
            result = chain.run(query)
        st.success("Answer:")
        st.write(result)

    except Exception as e:
        st.error("❌ Error processing transcript.")
        st.exception(e)
    finally:
        os.remove(tmp_file_path)
