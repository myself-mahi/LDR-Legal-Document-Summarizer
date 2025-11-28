import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pdfplumber
import torch
import re



# MPS DEVICE SETUP (Optimized for M4)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()



# Streamlit Config

st.set_page_config(page_title="Legal Document Summarizer", page_icon="⚖️", layout="wide")
st.title("⚖️ Legal Document Summarizer ")
st.markdown("Upload a **legal PDF** to generate a concise summary.")



# Sidebar Settings

st.sidebar.title("⚙️ Settings")
max_length = st.sidebar.slider("Max Summary Length", 150, 400, 250)
min_length = st.sidebar.slider("Min Summary Length", 50, 200, 100)
chunk_size = st.sidebar.slider("Chunk Size (words)", 300, 1200, 600)



# Load HuggingFace Model 

MODEL_NAME = "Arsomuu/Legal_T5"

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.to(device)   

    return tokenizer, model

tokenizer, model = load_model()



# PDF Extraction
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()



# Chunking

def chunk_text(text, max_words):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current, count = [], [], 0

    for sent in sentences:
        words = sent.split()
        if count + len(words) > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks



# Summarization
def summarize_chunk(chunk, min_len, max_len):
    encoding = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=1024
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **encoding,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def summarize_chunks(chunks, min_len, max_len):
    all_summaries = []
    progress = st.progress(0)
    placeholder = st.empty()

    for i, chunk in enumerate(chunks):

        try:
            summary = summarize_chunk(chunk, min_len, max_len)
        except Exception as e:
            summary = f"[Error in chunk {i+1}] {e}"

        all_summaries.append(summary)

        progress.progress((i + 1) / len(chunks))
        placeholder.markdown(f"### Chunk {i+1}/{len(chunks)}\n{summary}\n---")

    progress.empty()
    placeholder.empty()
    return "\n\n".join(all_summaries)



# File Upload + Processing

uploaded_file = st.file_uploader("📄 Upload Legal PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📥 Extracting text..."):
        full_text = extract_text_from_pdf(uploaded_file)

    if not full_text:
        st.error("❌ No readable text found. This PDF may be scanned.")
    else:
        st.success(f"Extracted {len(full_text.split())} words.")

        if st.button("✨ Generate Summary", use_container_width=True):
            chunks = chunk_text(full_text, chunk_size)

            with st.spinner("🤖 Summarizing using MPS on M4..."):
                final_summary = summarize_chunks(chunks, min_length, max_length)

            st.subheader("🧾 Final Summary")
            st.write(final_summary)

            st.download_button(
                "💾 Download Summary",
                data=final_summary,
                file_name="legal_summary.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    st.info("👆 Upload a PDF to start.")