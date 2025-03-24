import os
import torch
import fitz
import re
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
import pandas as pd
import base64
import io

def load_text_from_pdf(file):
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                try:
                    text += page.get_text()
                except Exception as e:
                    st.warning(f"Error on page {page.number}: {e}")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return ""
    return text

def remove_header_info(text):
    cleaned_text = text
    title_patterns = [
        r'^(.*?)(?=Abstract|ABSTRACT|Introduction|INTRODUCTION|\n\n)',
        r'^(.*?\d{4}.*?)(?=Abstract|ABSTRACT|Introduction|INTRODUCTION|\n\n)',
        r'^(.*?(?:University|Institute|Department|Email|@|Corresponding author).*?)(?=Abstract|ABSTRACT|Introduction|INTRODUCTION|\n\n)',
    ]
    for pattern in title_patterns:
        match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
        if match and len(match.group(1).split()) < 150:
            cleaned_text = cleaned_text.replace(match.group(1), '', 1)
            break
    return cleaned_text.strip()

def extract_paper_content(text):
    text = remove_header_info(text)
    abstract_patterns = [
        r'(?i)Abstract[\s\n]*\n(.*?)(?=\n\s*(?:Introduction|Keywords|1\.|\nI\.))',
        r'(?i)(?:^|\n)(?:\d+[\.\s]+)?Abstract[\s\n]+(.*?)(?=\n\s*(?:Introduction|Keywords|1\.|\nI\.))',
        r'(?i)Abstract[\s\n]+(.*?)(?=\n\n)',
        r'(?i)(?:^|\n)\\begin\{abstract\}(.*?)\\end\{abstract\}',
        r'ABSTRACT[\s\n]+(.*?)(?=\n\s*(?:Introduction|Keywords|1\.|\nI\.))',
        r'ABSTRACT[\s\n]+(.*?)(?=\n\n)'
    ]
    original_abstract = ""
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            original_abstract = match.group(1).strip()
            break
    main_content_markers = [
        r'(?i)Introduction\s*\n',
        r'(?i)1\.\s*Introduction',
        r'(?i)I\.\s*Introduction',
        r'(?i)Background',
        r'(?i)Related Work',
        r'(?i)Literature Review',
        r'(?i)Methodology',
        r'(?i)Proposed Method',
        r'(?i)Experimental Setup'
    ]
    main_content_start = len(text)
    for marker in main_content_markers:
        match = re.search(marker, text)
        if match and match.start() < main_content_start:
            main_content_start = match.start()
    reference_markers = [
        r'(?i)References\s*\n',
        r'(?i)Bibliography',
        r'(?i)Works Cited',
        r'(?i)Literature Cited',
        r'(?i)Acknowledgements',
        r'(?i)Acknowledgments'
    ]
    main_content_end = len(text)
    for marker in reference_markers:
        match = re.search(marker, text)
        if match and match.start() < main_content_end:
            main_content_end = match.start()
    main_content = text[main_content_start:main_content_end] if main_content_start < main_content_end else text
    return main_content, original_abstract

def preprocess_text(text, lemmatizer, stop_words, lemmatize=True, remove_stopwords=False):
    if not text:
        return ""
    try:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        text = re.sub(r'References\s+', '', text)
        text = re.sub(r'Acknowledgements\s+', '', text)
        if lemmatize or remove_stopwords:
            words = text.split()
            if remove_stopwords:
                words = [word for word in words if word.lower() not in stop_words]
            if lemmatize:
                try:
                    words = [lemmatizer.lemmatize(word) for word in words]
                except Exception as e:
                    st.warning(f"Error during lemmatization: {e}. Skipping lemmatization.")
            text = ' '.join(words)
        return text
    except Exception as e:
        st.warning(f"Error in preprocess_text: {e}")
        return text

def generate_abstract(model, tokenizer, device, text, lemmatizer, stop_words, max_length=256):
    paper_content, _ = extract_paper_content(text)
    max_input_tokens = 1024
    cleaned_content = re.sub(r'\s+', ' ', paper_content)
    cleaned_content = re.sub(r'\n+', ' ', cleaned_content)
    tokenized_input = tokenizer.encode(cleaned_content, add_special_tokens=False)
    if len(tokenized_input) > max_input_tokens - 20:
        tokenized_input = tokenized_input[:max_input_tokens - 20]
        cleaned_content = tokenizer.decode(tokenized_input)
    processed_input = preprocess_text(cleaned_content, lemmatizer, stop_words, lemmatize=True, remove_stopwords=False)
    prefix = "summarize: "
    input_ids = tokenizer.encode(prefix + processed_input, return_tensors="pt", truncation=True)
    input_ids = input_ids.to(device)
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    generated_abstract = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_abstract = post_process_abstract(generated_abstract)
    return generated_abstract, ""

def post_process_abstract(text):
    text = re.sub(r'^Abstract[\s\n]*:?\s*', '', text, flags=re.IGNORECASE)
    author_patterns = [
        r'\b(?:by|author[s]?)\s*:.*?(?=\n|\Z)',
        r'(?:university|institute).*?(?=\n|\Z)',
        r'(?:department).*?(?=\n|\Z)',
        r'email:.*?(?=\n|\Z)',
        r'corresponding author:.*?(?=\n|\Z)'
    ]
    for pattern in author_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_download_link(text, filename, label="Download Results"):
    buffer = io.BytesIO()
    buffer.write(text.encode())
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 {label}</a>'
    return href

def run_abstract_generator(model, tokenizer, device, model_path, lemmatizer, stop_words, uploaded_file=None):
    st.title("Research Paper Abstract Generator")
    st.markdown("""
    This tool automatically generates high-quality abstracts for research papers using a fine-tuned T5 model.
    Upload your PDF files, and get concise, informative abstracts in seconds.
    """)
    st.info(f"Using model from: {model_path} on {device}")

    if uploaded_file is None:
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    else:
        uploaded_files = [uploaded_file]

    if not uploaded_files:
        st.info("Please upload one or more PDF files to continue.")
        return

    st.subheader("Generation Options")
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum abstract length (words):", min_value=50, max_value=500, value=200, step=10)
    with col2:
        format_option = st.radio("Format output as:", options=["Plain text", "Markdown", "HTML"], index=0)

    if st.button("Generate Abstracts", type="primary", use_container_width=True):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            try:
                with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                    text = load_text_from_pdf(uploaded_file)
                if not text:
                    results.append({"filename": uploaded_file.name, "status": "Error: No text extracted", "abstract": "", "original_abstract": ""})
                    continue
                with st.spinner(f"Generating abstract for {uploaded_file.name}..."):
                    generated_abstract, _ = generate_abstract(model, tokenizer, device, text, lemmatizer, stop_words, max_length=max_length)
                results.append({"filename": uploaded_file.name, "status": "Success", "abstract": generated_abstract, "original_abstract": ""})
            except Exception as e:
                results.append({"filename": uploaded_file.name, "status": f"Error: {str(e)}", "abstract": "", "original_abstract": ""})
            progress_bar.progress((i + 1) / len(uploaded_files))
        progress_bar.empty()
        status_text.empty()

        st.subheader("Generated Abstracts")
        download_text = ""
        for result in results:
            with st.expander(f"{result['filename']} - {result['status']}"):
                if result['status'] == "Success":
                    st.markdown("**Generated Abstract:**")
                    st.markdown(result['abstract'])
                    st.code(result['abstract'], language="text")
                else:
                    st.error(result['status'])
            download_text += f"File: {result['filename']}\nStatus: {result['status']}\nGenerated Abstract:\n{result['abstract']}\n\n{'='*70}\n\n"

        results_df = pd.DataFrame([
            {"File": r["filename"], "Status": r["status"], "Generated Abstract": r["abstract"][:100] + "..." if len(r["abstract"]) > 100 else r["abstract"]}
            for r in results
        ])
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(get_download_link(download_text, "generated_abstracts.txt", "Download as Text File"), unsafe_allow_html=True)
        with col2:
            csv = results_df.to_csv(index=False)
            st.markdown(get_download_link(csv, "generated_abstracts.csv", "Download as CSV"), unsafe_allow_html=True)

def main():
    st.warning("This module should be run via main.py")

if __name__ == "__main__":
    main()