import warnings
warnings.filterwarnings('ignore')
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

# Hugging Face authentication
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("Hugging Face token not found! Set HF_TOKEN in environment variables.")
    st.stop()

# Initialize Hugging Face Inference Client
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)

def process_pdf(uploaded_file):
    try:
        if uploaded_file is None:
            return "No file uploaded."

        # Save the uploaded file temporarily
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        summaries = []  # 
        for chunk in chunks[:5]:  # Process first 5 chunks only
            text = " ".join(chunk.page_content.split()[:500])  # First 500 words
            response = client.text_generation(
                prompt=f"Summarize this text:\n\n{text}\n\nSummary:",
                max_new_tokens=150,  # Shorter summary length
                stream=False  # Ensure synchronous response
            )
            summaries.append(response.strip())

        return "\n\n".join(summaries)  # ✅ Properly return the summarized output

    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Document Summarization Chatbot ")
st.write("Upload a PDF document and get a summarized version.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Button to process the PDF
if st.button("Summarize"):
    if uploaded_file:
        with st.spinner("Summarizing..."):
            summary = process_pdf(uploaded_file)
        st.text_area("Summary", summary, height=300)
    else:
        st.warning("Please upload a PDF file first.")
