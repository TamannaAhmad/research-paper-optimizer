import streamlit as st
import tempfile
import os
import torch
import PyPDF2
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
import import_ipynb

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Research Paper Optimizer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules from separate files after setting page config
from citation_validator import run_citation_validator
from image_caption_generator import run_image_captioning
from abstract_generator import run_abstract_generator
from quality_assurance import run_quality_assurance
from relevance_checker import run_relevance_checker

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
        # Check if model path is a Hugging Face model ID (contains '/')
        is_huggingface_model = '/' in model_path
        
        st.info(f"Loading model from: {'Hugging Face' if is_huggingface_model else 'local path'} - {model_path}")
        
        # Load tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # Move model to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        st.success(f"Model successfully loaded from {model_path}!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None, None, None

# Find a valid model path
def find_valid_model_path():
    # Check environment variable first
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        path_obj = Path(env_path)
        if path_obj.exists() and ((path_obj / "pytorch_model.bin").exists() or (path_obj / "config.json").exists()):
            return str(path_obj.absolute())
    
    # If Hugging Face model is specified, prioritize it
    hf_model = "srii19/abstract_generator_model"
    
    # Check if the model is a Hugging Face model by looking for '/'
    if '/' in hf_model:
        return hf_model  # Return the HF model ID directly
    
    # Check local paths as fallback
    local_paths = [
        "./models/final_model",
        "./fine_tuned_model",
        "./model",
        "./results/final_model",
        "../results/final_model",
        "./models/fine_tuned_model",
        "./saved_models/fine_tuned_model",
        "./output/model",
        "./checkpoints/final",
    ]
    
    for path in local_paths:
        path_obj = Path(path)
        if path_obj.exists() and ((path_obj / "pytorch_model.bin").exists() or (path_obj / "config.json").exists()):
            return str(path_obj.absolute())
    
    # If no valid path found but we have a HF model, return that
    if hf_model:
        return hf_model
        
    return None

def display_home():
    st.header("Welcome to the Research Paper Optimizer Tool")
    st.markdown("This application helps researchers in their research paper writing process using the following tools:")
    # Display feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Abstract Generator")
        st.write("Generate concise summaries of research papers, useful for writing abstracts.")
        st.subheader("Image Captioning")
        st.write("Generates relevant captions for scientific figures.")
        st.subheader("Citation Validator")
        st.write("Checks if references are correctly cited and marked within the text.")
    
    with col2:
        st.subheader("Quality Assurance")
        st.write("Ensures the paper's contents are logically coherent and grammatically correct.")
        st.subheader("Relevance Checker")
        st.write("Determines if a reference is relevant to your research paper.")
    
    # Instructions
    st.header("How to Use")
    st.write("""
    1. Upload a PDF research paper using the file uploader in the sidebar
    2. Select the tool you want to use from the navigation menu
    3. Configure the tool settings as needed
    4. View the results
    """)

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
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Abstract Generator", "Image Captioning", "Citation Validator", "Quality Assurance", "Relevance Checker"])
    
    # File uploader in sidebar
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a research paper (PDF)", type="pdf")
    
    # Pass resources and file to tools
    with tab0:
        display_home()
    with tab1:
        run_abstract_generator(model, tokenizer, device, model_path, lemmatizer, stop_words, uploaded_file=uploaded_file)
    with tab2:
        run_image_captioning(uploaded_file)
    with tab3:
        run_citation_validator(uploaded_file)
    with tab4:
        run_quality_assurance(uploaded_file)
    with tab5:
        run_relevance_checker(uploaded_file)
    
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
    st.markdown("© 2025 Research Paper Optimizer")

if __name__ == "__main__":
    main()