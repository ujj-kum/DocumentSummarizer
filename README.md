# Document Summarization Chatbot  

A Streamlit-based web app that summarizes PDFs using Mistral-7B-Instruct-v0.2 via Hugging Face API.  
```Currently working for PDFs only```

## Features  
- PDF Upload & Parsing (LangChain & PyPDFLoader)  
- Text Splitting (RecursiveCharacterTextSplitter)  
- AI-Powered Summarization (Mistral-7B-Instruct-v0.2)  
- Interactive UI (Streamlit)  

## Tech Stack  
- Model: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)  
- Frameworks: Streamlit, LangChain, Hugging Face API  

## Setup  

### Install Dependencies  
```sh
pip install -r requirements.txt
