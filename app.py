import streamlit as st
import PyPDF2
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set page configuration
st.set_page_config(page_title="AI Study Assistant", layout="wide")

# Set environment variables to reduce model loading memory usage
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cache the models to avoid reloading
@st.cache_resource
def load_models():
    st.info("Loading NLP models - this may take a minute...")
    
    # Load TinyLlama model with reduced precision
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        low_cpu_mem_usage=True,
        torch_dtype="auto"
    )
    
    # Load smaller summarization model
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn", 
        tokenizer="facebook/bart-large-cnn",
        device="cpu"
    )
    
    return tokenizer, model, summarizer

# Document processing functions
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    # Process only first 10 pages for speed
    max_pages = min(len(pdf_reader.pages), 10)
    for i in range(max_pages):
        text += pdf_reader.pages[i].extract_text() + "\n"
    
    if len(pdf_reader.pages) > max_pages:
        text += f"\n[Note: Only {max_pages} of {len(pdf_reader.pages)} pages processed]"
    
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    else:
        st.error("Unsupported file type. Please upload PDF or text files.")
        return None

# AI processing functions
def generate_answer(tokenizer, model, question, context=None):
    with st.spinner("Generating answer..."):
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer based on the context."
        else:
            full_prompt = question
            
        formatted_prompt = f"[INST] {full_prompt} [/INST]"
        
        # Handle long context by taking portions from beginning and end
        if context and len(context) > 2000:
            context = context[:1000] + "\n...\n" + context[-1000:]
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        output = model.generate(
            **inputs, 
            max_new_tokens=150,
            do_sample=True, 
            top_p=0.9, 
            top_k=30,
            temperature=0.7,
            num_beams=1
        )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response.replace(formatted_prompt, "").strip()

def generate_summary(summarizer, text, max_length=150, min_length=40):
    with st.spinner("Generating summary..."):
        # Handle very long texts by processing key sections
        if len(text) > 10000:
            processed_text = text[:3000] + "\n...\n" + text[-3000:]
        else:
            processed_text = text
        
        summary = summarizer(
            processed_text, 
            max_length=max_length, 
            min_length=min_length, 
            do_sample=False,
            truncation=True
        )[0]['summary_text']
        
        return summary

# Main UI
st.title("🎓 AI Study Assistant")
st.subheader("Upload Documents, Ask Questions, or Summarize Text")

# Sidebar
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or Text file", type=['pdf', 'txt'])
    
    document_text = None
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            document_text = extract_text_from_file(uploaded_file)
            
        if document_text:
            st.success(f"Uploaded: {uploaded_file.name}")
            st.text("Document preview:")
            preview_text = document_text[:200]
            if len(document_text) > 200:
                preview_text += "..."
            st.text(preview_text)
    
    # Load models button
    if st.button("Load AI Models"):
        tokenizer, model, summarizer = load_models()
        st.success("Models loaded successfully!")

# Load the models at startup for simplicity
try:
    tokenizer, model, summarizer = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {str(e)}")

# Tabs for different features
tab1, tab2= st.tabs(["Ask a Question", "Summarize Custom Text"])

with tab1:
    st.header("Ask Anything")
    if document_text:
        st.info("You can ask questions about the uploaded document.")
    
    question = st.text_input("Your question:")
    if question:
        if st.button("Get Answer", key="ask_button"):
            if models_loaded:
                try:
                    answer = generate_answer(tokenizer, model, question, document_text)
                    st.markdown("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Models not loaded. Please load the AI models first.")

with tab2:
    st.header("Summarize Custom Text")
    st.info("Paste text here if you want to summarize it.")
    custom_text = st.text_area("Enter text to summarize:")
    
    if custom_text:
        max_length = st.slider("Maximum summary length (words)", 50, 300, 150, key="custom_max")
        min_length = st.slider("Minimum summary length (words)", 20, 100, 40, key="custom_min")
        
        if st.button("Generate Summary", key="custom_summary_button"):
            if models_loaded:
                try:
                    summary = generate_summary(summarizer, custom_text, max_length, min_length)
                    st.markdown("**Summary:**")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
            else:
                st.warning("Models not loaded. Please load the AI models first.")

# Footer
st.markdown("---")
st.caption("AI Study Assistant - Powered by TinyLlama and BART")