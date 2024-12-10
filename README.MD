
![image](https://github.com/ahmadalis2016/Iridium-AI-Image-Analysis/assets/130319416/3590b637-b72a-4a41-86ba-a556e9c22016) 

## Iridium AI: Chat with PDF 

This Streamlit application allows users to interactively chat with the content of PDF files using the Gemini conversational AI model. Users can upload one or more PDF files, and then ask questions related to the content of those files. The Gemini model will provide answers based on the provided context.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iridium-ai-llama2chatbot.streamlitapp.com/)

## Prerequisite libraries

```
streamlit
google-generativeai
python-dotenv
langchain
PyPDF2
chromadb
faiss-cpu
langchain_google_genai
```

## Overview
The Llama 2 Chatbot is an interactive chatbot application developed using Streamlit, a popular Python library for building web applications. It leverages the advanced language generation capabilities of the Llama 2 LLM (Large Language Model) model developed by Meta (formerly Facebook).

## Features
Upload PDF Files: Users can upload one or more PDF files containing the content they want to analyze.

Text Extraction: The application extracts text from the uploaded PDF files to create a text corpus for analysis.

Vector Store Creation: It generates a vector store using the extracted text for efficient similarity search.

Conversational AI: Users can ask questions related to the content of the PDF files, and the Gemini model will provide answers based on the context from the uploaded files.


## Usage
Upload PDF files: Use the file uploader in the sidebar to upload one or more PDF files containing the content you want to analyze.

Submit & Process: Click the "Submit & Process" button to extract text from the uploaded PDF files and create a vector store for similarity search.

Ask a Question: Enter your question in the text input field and press Enter to submit it. The Gemini model will generate a response based on the provided context from the PDF files.

## Contributing
Ahmad A Shaik, Ph. D.
 
## License
This project is licensed under the MIT License. 
