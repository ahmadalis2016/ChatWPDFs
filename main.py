import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PIL import Image
import os
from difflib import HtmlDiff

# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(role):
    if role == "User":
        prompt_template = """
        You are assisting a user. Answer their questions concisely and informatively based on the context provided. 
        If the answer is not in the context, say, "The answer is not available in the provided context."
        
        Context:\n {context}\n
        Question:\n {question}\n
        Answer:
        """
    elif role == "Service Desk Personnel":
        prompt_template = """
        You are acting as a Service Desk Personnel. Provide detailed and actionable answers based on the context provided. 
        If the answer is not in the context, offer guidance on possible next steps or sources of information.
        
        Context:\n {context}\n
        Question:\n {question}\n
        Answer:
        """
    else:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer is not available in the provided context, just say, "Answer is not available in the context."
        
        Context:\n {context}\n
        Question:\n {question}\n
        Answer:
        """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_pdf_files(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

def get_user_input(user_question, role):
    faiss_index_path = "faiss_index/index.faiss"
    if not os.path.exists(faiss_index_path):
        st.error("FAISS index file not found. Make sure to upload PDF files and process them first.")
        return ""

    new_db = FAISS.load_local("faiss_index", embeddings=GoogleGenerativeAIEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"), model="models/embedding-001"), allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(role)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def show_survey():
    st.subheader("Survey: Feedback on ChatPDF Application")
    with st.form("survey_form"):
        q1 = st.radio(
            "How easy was it to upload and process PDFs for chat?",
            ["Very Easy", "Easy", "Neutral", "Difficult", "Very Difficult"],
            index=0
        )
        q2 = st.radio(
            "Did the responses align with your expectations based on the role (User/Service Desk Personnel)?",
            ["Yes, completely", "Mostly", "Somewhat", "Not much", "Not at all"],
            index=0
        )
        q3 = st.text_area(
            "What additional features or improvements would you like to see in the application?",
            placeholder="Enter your suggestions here..."
        )
        submitted = st.form_submit_button("Submit Survey")
        if submitted:
            st.success("Thank you for your feedback!")
            st.write("Your responses:")
            st.write(f"1. Upload/Process Ease: {q1}")
            st.write(f"2. Role Response Alignment: {q2}")
            st.write(f"3. Suggestions: {q3}")

def compare_documents_with_highlight(doc1_text, doc2_text):
    diff = HtmlDiff().make_file(
        doc1_text.splitlines(),
        doc2_text.splitlines(),
        "Document 1",
        "Document 2"
    )
    return diff

def main():
    st.set_page_config(page_title="ChatPDF", layout="wide")
    st.header("ChatPDF : AI-Driven Conversation")

    # Sidebar for role selection and file upload
    with st.sidebar:
        logo_path = "Images/IridiumAILogo.png"
        iridium_logo = Image.open(logo_path)
        st.image(iridium_logo, use_container_width=False)

        st.title("Menu:")
        role = st.selectbox("Select Your Role:", ["User", "Service Desk Personnel"])
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing PDF documents..."):
                process_pdf_files(pdf_docs)
                st.success("PDFs processed successfully.")

        if st.button("Take Survey"):
            show_survey()

    # Main chat interface
    if role:
        user_question = st.text_input(f"({role}) Ask a Question from the PDF Files")
        if user_question:
            st.write(f"{role}'s Reply:")
            st.write(get_user_input(user_question, role))

    # Document Comparison
    st.subheader("Document Comparison")
    comparison_docs = st.file_uploader("Upload Two PDFs for Comparison", accept_multiple_files=True, key="comparison_uploader")
    if comparison_docs and len(comparison_docs) == 2:
        with st.spinner("Comparing documents..."):
            doc1_text = get_pdf_text([comparison_docs[0]])
            doc2_text = get_pdf_text([comparison_docs[1]])
            highlighted_diff = compare_documents_with_highlight(doc1_text, doc2_text)
            st.components.v1.html(highlighted_diff, height=600, scrolling=True)

if __name__ == "__main__":
    main()
