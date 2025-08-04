import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

# Set up Streamlit UI
st.set_page_config(page_title="🦜🔍 Fireflies Market Assistant", layout="wide")
st.title("🦜🔍 Fireflies Market Assistant")
st.markdown(
    "Upload a Fireflies meeting transcript and get market intelligence insights using GPT."
)

# Upload .txt file
uploaded_file = st.file_uploader("Upload a Fireflies transcript (.txt)", type=["txt"])
user_question = st.text_input("Ask a specific question (optional):")

# Load API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

if uploaded_file:
    # Save file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load document
    loader = TextLoader(tmp_path, encoding="utf-8")
    texts = loader.load()

    # Split long documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(texts)

    # Create embeddings and vector DB
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)

    # Create retriever-based QA system
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # Run chain
    query = user_question if user_question else "What are the key market insights?"
    with st.spinner("Analyzing transcript..."):
        result = qa_chain({"query": query})
        answer = result["result"]
        st.subheader("🧠 Answer")
        st.write(answer)

    # Clean up
    os.remove(tmp_path)
