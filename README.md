<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Streamlit-red?style=for-the-badge" alt="Made with Streamlit">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Model-TinyLlama%20%26%20BART-green?style=for-the-badge" alt="Model TinyLlama and BART">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Project Status Active">
</p>

# AI Study Assistant 🎓🤖

An AI-powered Study Assistant built with **Streamlit**, **TinyLlama**, and **BART**, designed to help students and professionals quickly extract summaries and generate answers from academic documents.  
This tool streamlines document comprehension, enhances study efficiency, and provides real-time AI support for learning and research purposes.

[🔗 Demo Link](https://chat-with-pdf-ms.streamlit.app/)

---

## ✨ Features

- 📄 Upload and process **PDF** or **Text** files
- 🔍 **Ask context-aware questions** from the uploaded documents
- 📝 **Summarize** uploaded document text or custom text
- ⚡ **Lightweight models** ensure fast and responsive performance
- 🔁 **Cached models** to prevent reloading and speed up response
- 🛡️ **Memory optimized** for smoother performance

---

## 📋 How It Works

### Document Processing
- **PDF Files**: Extracts text from the first 10 pages for better speed.
- **Text Files**: Reads complete text content.
- **Unsupported Files**: Displays an error message.

### Question Answering
- Uses **TinyLlama-1.1B-Chat-v1.0** to generate natural-sounding answers.
- Context is intelligently truncated if too large (>2000 characters).
- Sampling techniques (`top_p`, `top_k`, `temperature`) make the responses more human-like.

### Summarization
- Uses **facebook/bart-large-cnn** to generate concise summaries.
- Handles very large texts by smartly trimming the input.
- Adjustable `max_length` and `min_length` parameters for custom summary size.

---

## 🏗️ Architecture Overview

```
User Uploads Document (PDF/TXT)
        ↓
Extract Text from Document
        ↓
[Option 1] Ask Question → TinyLlama Generates Answer
[Option 2] Summarize Text → BART Summarizes
        ↓
Display Results
```

---

## 🛠️ Installation and Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-study-assistant.git
   cd ai-study-assistant
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📦 Requirements

- streamlit
- PyPDF2
- transformers
- torch

*(All dependencies are listed in `requirements.txt`.)*

---

## 📈 Optimization Techniques

- **Model Caching**: Models are cached using `@st.cache_resource` to speed up reruns.
- **Memory Optimization**: `low_cpu_mem_usage=True` and `torch_dtype="auto"` are used to reduce resource usage.
- **Limited Page Processing**: For PDFs, only the first 10 pages are processed to maintain responsiveness.

---

## 🚀 Future Improvements

- Full-document processing for larger PDFs
- OCR integration for scanned PDFs
- Additional summarization and QA models (like GPT-2, T5)
- DOCX file support
- Multi-language support

---

## 👨‍💻 Authors

- Mainak Sen
- Yukta Bhardwaj
- G. Tarunika

---

## 📜 License

This project is for educational and demonstration purposes.  
Feel free to extend and modify — just give appropriate credit!

---
