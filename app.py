import warnings
warnings.filterwarnings('ignore')
import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from huggingface_hub import login
import sys
import torch
__import__('pysqlite3')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

hf_token = os.getenv("HF_TOKEN")  # Ensure the token is loaded

if hf_token:
    login(token=hf_token)  # Authenticate with Hugging Face
    print("#"*100)
    print("LOGIN SUCCESSFUL!!!")
    print("#"*100)
else:
    print("#"*100)
    print('NOT FOUND!!!!!!!')
    print("#"*100)
    
## Load Hugging Face LLM
@st.cache_resource()
def get_llm():
    hf_pipeline = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        token=hf_token,  # Make sure this is set in the environment
        device_map=device
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


## Load Hugging Face Embeddings
@st.cache_resource()
def huggingface_embedding():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


## Load PDF and Process
def process_pdf(uploaded_file):
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Save the uploaded file to a temporary location
            temp_path = "temp_uploaded.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load the PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)

            embedding_model = huggingface_embedding()
            vectordb = Chroma.from_documents(chunks, embedding_model)
            return vectordb
    return None


## QA Retrieval Function
def retriever_qa(vectordb, query):
    llm = get_llm()
    retriever_obj = vectordb.as_retriever()
    
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                     chain_type="stuff", 
                                     retriever=retriever_obj, 
                                     return_source_documents=False)

    response = qa.invoke(query)
    return response['result']


# Streamlit UI
st.set_page_config(page_title="RAG Chatbot (Hugging Face)", layout="wide")
st.title("📄 RAG Chatbot (Hugging Face)")
st.write("Upload a PDF document and ask any question. The chatbot will use the document to answer.")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'])

if uploaded_file is not None:
    vectordb = process_pdf(uploaded_file)

    if vectordb:
        query = st.text_input("Ask a question based on the document:")
        
        if query:
            with st.spinner("Retrieving answer..."):
                answer = retriever_qa(vectordb, query)
                st.success(answer)
