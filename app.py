import os
from dotenv import load_dotenv
import streamlit as st
import pdfplumber
from transformers import pipeline
import asyncio
import base64


# async loop handling

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables form .env file
load_dotenv()

# Fetch token
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# Cache the summarizer model

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", token=hf_token)

summarizer = load_summarizer()


# EXTRACT text from pdf 
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:     # <-- prevents NoneType error
                text += page_text + "\n"
    return text if text.strip() else "[ERROR] No text found in the pdf. Please try another File."


def chunk_text(text, max_length=1500):
    chunks = []
    for i in range(0, len(text), max_length):
        chunks.append(text[i:i+max_length])
    return chunks


# Web designing using html and css

st. markdown(
    """
    <style>
/* ---- Global Background ---- */
    body {
        background: linear-gradient(135deg, #e3f2fd, #ffffff);
        font-family: 'Segoe UI', sans-serif;
    }

    /* ---- Main container ---- */
    .main {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.1);
    }

    /* ---- Title ---- */
    h1 {
        color: #0056D2;
        text-align: center;
        font-size: 40px !important;
        font-weight: 700;
    }

    /* ---- File uploader box ---- */
    .stFileUploader {
        border: 2px dashed #0056D2 !important;
        border-radius: 15px;
        padding: 15px;
    }

    /* ---- Text box ---- */
    .stTextArea, .stTextInput>div>input {
        border-radius: 10px !important;
        border: 1px solid #b9d6ff !important;
    }

    /* ---- Buttons ---- */
    .stButton>button {
        background-color: #0056D2;
        color: white;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 18px;
        width: 100%;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #003f9a;
        transform: scale(1.03);
    }
</style>
""", unsafe_allow_html=True)

# STREAMLIT UI

st.markdown("""
<div style="text-align:center; padding: 20px;">
    <h1 style="color:#0056D2; font-size: 42px; font-weight:700;">
         Indian Legal Document Summarizer
    </h1>
    <p style="font-size:18px; color:#343A40;">
        Upload a legal document (PDF) and get a concise summary powered by AI.
    </p>
</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Drag & Drop or Browse", type=["pdf"], help="Upload A PDF file for summarization")

if uploaded_file:
    with st.spinner("Extracting text from PDF...."):
        text = extract_text_from_pdf(uploaded_file)
    
    if text.startswith("[ERROR]"):
        st.error(text)

    else:
        st.success("Text extracted successfully! Generating summary...")

        #  Split into chunks (no variable rename)
        chunks = chunk_text(text)
        summary = ""  # ← keep same variable name

        with st.spinner("Summarizing..."):
            for i, chunk in enumerate(chunks):
                st.write(f"🔄 Summarizing section {i+1}/{len(chunks)}...")
                part_summary = summarizer(
                    chunk,
                    max_length=200,
                    min_length=80,
                    do_sample=False
                )[0]['summary_text']

                summary += part_summary + "\n\n"   # add to final summary

        st.markdown("### ✅ Summary:")
        st.info(summary)


        # ---- DOWNLOAD BUTTON ----
        summary_bytes = summary.encode('utf-8')
        b64 = base64.b64encode(summary_bytes).decode()

        href = f"""
        <a href="data:file/txt;base64,{b64}" 
        download="summary.txt"
        style="text-decoration:none;">
            <button style="
                background-color:#0056D2;
                color:white;
                padding:10px 20px;
                border:none;
                border-radius:8px;
                font-size:16px;
                cursor:pointer;">
                📥 Download Summary
            </button>
        </a>
        """

        st.markdown(href, unsafe_allow_html=True)
