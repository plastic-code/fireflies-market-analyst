import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os

# Load OpenAI API Key securely from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# Streamlit UI
st.set_page_config(page_title="Fireflies Market Assistant", layout="wide")
st.title("🦜🔍 Fireflies Market Assistant")
st.markdown("Upload a Fireflies meeting transcript and get market intelligence insights using GPT.")

# Upload file
uploaded_file = st.file_uploader("Upload a Fireflies transcript (.txt)", type=["txt"])

# Optional: User question
query = st.text_input("Ask a specific question (optional):")

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp_transcript.txt", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split
    loader = TextLoader("temp_transcript.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Embedding & Vector Store
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    # Search
    if query:
        docs = db.similarity_search(query)
    else:
        docs = texts[:3]  # Default: use first few chunks

    # Run QA chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query or "What are the key market insights from this meeting?")

    st.subheader("📌 Extracted Insight")
    st.write(response)
