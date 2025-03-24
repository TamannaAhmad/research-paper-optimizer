import streamlit as st
import tempfile
import os
import torch
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Research Paper Optimizer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules from separate files after setting page config
from citation_validator import run_citation_validator
from abstract_generator import run_abstract_generator

# Initialize NLTK resources
@st.cache_resource
def initialize_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        return WordNetLemmatizer(), set(stopwords.words('english'))
    except Exception as e:
        st.warning(f"NLTK resource initialization error: {e}")
        stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                          'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those'])
        class FallbackLemmatizer:
            def lemmatize(self, word):
                return word
        return FallbackLemmatizer(), stop_words

# Load the fine-tuned model
@st.cache_resource
def load_model(model_path):
    try:
        st.info(f"Attempting to load model from: {model_path}")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None, None, None

# Find a valid model path
def find_valid_model_path():
    POTENTIAL_MODEL_PATHS = [
        os.environ.get("MODEL_PATH"),
        r"research-paper-analyser/results/final_model",
        "./fine_tuned_model",
        "./model",
        "./results/final_model",
        "../results/final_model",
        "./models/fine_tuned_model",
        "./saved_models/fine_tuned_model",
        "./output/model",
        "./checkpoints/final",
    ]
    paths_to_check = [p for p in POTENTIAL_MODEL_PATHS if p is not None]
    for path in paths_to_check:
        path_obj = Path(path)
        if path_obj.exists() and ((path_obj / "pytorch_model.bin").exists() or (path_obj / "config.json").exists()):
            return str(path_obj.absolute())
    return None

def main():
    st.title("Research Paper Optimization Tools")

    # Initialize NLTK resources
    lemmatizer, stop_words = initialize_nltk()

    # Load model
    model_path = find_valid_model_path()
    if model_path:
        model, tokenizer, device = load_model(model_path)
        if model is None:
            st.error("Failed to load the T5 model. Please ensure the model files are in the specified path.")
            return
    else:
        st.error("No valid model path found. Please specify the path to your fine-tuned T5 model.")
        model_path = st.text_input("Enter model path manually:", "")
        if model_path and st.button("Load Model"):
            model, tokenizer, device = load_model(model_path)
            if model is None:
                st.error("Model loading failed. Please check the path and try again.")
                return
        else:
            return

    # Create tabs for different tools
    tab1, tab2 = st.tabs(["Abstract Generator", "Citation Validator"])

    # File uploader in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload a research paper (PDF)", type="pdf")

    # Pass resources and file to tools
    with tab1:
        run_abstract_generator(model, tokenizer, device, model_path, lemmatizer, stop_words, uploaded_file=uploaded_file)
    with tab2:
        run_citation_validator(uploaded_file=uploaded_file)

    # Display file details if uploaded
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size / 1024:.2f} KB"}
        st.sidebar.write("File Details:")
        for key, value in file_details.items():
            st.sidebar.write(f"- {key}: {value}")
        os.unlink(pdf_path)

    # Footer
    st.markdown("---")
    st.markdown("© 2025 Academic Paper Tools")

if __name__ == "__main__":
    main()