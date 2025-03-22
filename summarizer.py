import os
import re
import pandas as pd
import torch
import fitz  # PyMuPDF
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Download ALL required NLTK resources
def download_nltk_resources():
    resources = [
        'punkt',
        'wordnet',
        'stopwords',
        'punkt_tab' 
    ]
    
    for resource in resources:
        try:
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Failed to download {resource}: {e}")
            # Continue with other resources even if one fails

# Download resources
download_nltk_resources()

# Initialize tokenizer and preprocessing tools globally
tokenizer = T5Tokenizer.from_pretrained("t5-small")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load dataset from PDF with improved error handling
def load_dataset_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                try:
                    text += page.get_text()
                except Exception as e:
                    print(f"Error on page {page.number} in {file_path}: {e}")
                    # Continue with other pages
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
        return ""
    return text

# Improved abstract extraction with more patterns
def extract_abstract(text):
    # Common patterns for abstract sections in research papers (expanded)
    abstract_patterns = [
        # Pattern 1: Standard "Abstract" followed by text until Introduction
        r'(?i)Abstract[\s\n]*\n(.*?)(?=\n\s*(?:Introduction|Keywords|1\.|\nI\.))',
        
        # Pattern 2: Abstract with heading number
        r'(?i)(?:^|\n)(?:\d+[\.\s]+)?Abstract[\s\n]+(.*?)(?=\n\s*(?:Introduction|Keywords|1\.|\nI\.))',
        
        # Pattern 3: Abstract section without clear ending (use paragraph boundary)
        r'(?i)Abstract[\s\n]+(.*?)(?=\n\n)',
        
        # Pattern 4: ArXiv style abstract
        r'(?i)(?:^|\n)\\begin\{abstract\}(.*?)\\end\{abstract\}',
        
        # Pattern 5: Look for "ABSTRACT" in all caps
        r'ABSTRACT[\s\n]+(.*?)(?=\n\s*(?:Introduction|Keywords|1\.|\nI\.))',
        
        # Pattern 6: Look for ABSTRACT label without clear ending
        r'ABSTRACT[\s\n]+(.*?)(?=\n\n)'
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # Remove the abstract from the rest of the text
            remaining_text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
            print(f"Found abstract using pattern: {pattern[:30]}...")
            return abstract, remaining_text
    
    # Heuristic approach: Look for text after title/authors but before main content
    lines = text.split('\n')
    potential_abstract = ""
    
    # Try to find abstract by location (usually within first 15-20% of document)
    doc_start = "\n".join(lines[:int(len(lines) * 0.2)])  # First 20% of document
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', doc_start) if p.strip()]
    
    # Look for paragraph with reasonable length (not too short, not too long)
    # Skip very short paragraphs (likely titles or authors)
    for p in paragraphs:
        if 100 <= len(p) <= 1500 and not p.startswith('#') and not p.startswith('Figure'):
            potential_abstract = p
            print("Found potential abstract using heuristic approach")
            break
    
    if potential_abstract:
        # Remove the abstract from the full text
        remaining_text = text.replace(potential_abstract, "", 1)
        return potential_abstract, remaining_text
    
    print(f"Warning: No abstract found in document, using first 200 characters as placeholder")
    if len(text) > 400:
        return text[:200], text[200:]  # Use beginning as abstract if nothing else found
    else:
        return "", text  # Document too short, return empty abstract

# Modified text preprocessing with error handling
def preprocess_text(text, lemmatize=True, remove_stopwords=False):
    if not text:
        return ""
        
    try:
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with space
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove citations like [1], [2, 3], etc.
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        
        # Remove common paper sections if remaining
        text = re.sub(r'References\s+', '', text)
        text = re.sub(r'Acknowledgements\s+', '', text)
        
        if lemmatize or remove_stopwords:
            # Use simple word splitting instead of nltk tokenizer to avoid potential errors
            words = text.split()
            
            if remove_stopwords:
                words = [word for word in words if word.lower() not in stop_words]
                
            if lemmatize:
                try:
                    words = [lemmatizer.lemmatize(word) for word in words]
                except Exception as e:
                    print(f"Error during lemmatization: {e}. Skipping lemmatization.")
                    # Continue without lemmatization
                
            text = ' '.join(words)
        
        return text
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return text  # Return original text if processing fails

# Create dataset with abstract as target and rest as input
def create_dataset(paper_texts, max_input_length=8192, max_target_length=512):
    input_texts = []
    target_texts = []
    
    for i, text in enumerate(paper_texts):
        try:
            abstract, remaining_text = extract_abstract(text)
            
            if abstract and remaining_text:  # Only use papers where we found both abstract and content
                # Preprocess both abstract and remaining text
                processed_abstract = preprocess_text(abstract, lemmatize=True, remove_stopwords=False)
                processed_remaining = preprocess_text(remaining_text, lemmatize=True, remove_stopwords=False)
                
                if processed_abstract and processed_remaining:
                    input_texts.append(processed_remaining)
                    target_texts.append(processed_abstract)
                    print(f"Successfully processed document {i+1}")
        except Exception as e:
            print(f"Error processing document {i+1}: {e}")
            continue
    
    # Create a DataFrame with input and target texts
    df = pd.DataFrame({"input_text": input_texts, "target_text": target_texts})
    return Dataset.from_pandas(df)

# Tokenize function for T5 (handles both inputs and targets)
def tokenize_function(examples):
    # T5 uses a "prefix" for tasks - we'll use "summarize: " as our task prefix
    model_inputs = tokenizer(
        ["summarize: " + txt for txt in examples["input_text"]],
        max_length=1024,  # T5-small can handle up to 512, but we'll truncate longer inputs
        padding="max_length",
        truncation=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=256,  # Abstracts are usually shorter
            padding="max_length",
            truncation=True
        )
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load and preprocess dataset with better error handling
def load_and_preprocess_dataset(file_paths):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    all_texts = []
    for i, file_path in enumerate(file_paths):
        try:
            print(f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
            text = load_dataset_from_pdf(file_path)
            if text:
                all_texts.append(text)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    print(f"Successfully loaded {len(all_texts)} documents")
    
    # Create dataset with abstract as target and rest as input
    dataset = create_dataset(all_texts)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Set the format for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_dataset

# Fine-tune model
def fine_tune_model(train_dataset, val_dataset, output_dir):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,  # Further reduced batch size for safety
        per_device_eval_batch_size=2,
        num_train_epochs=8,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_accumulation_steps=4,  # Increased gradient accumulation steps
        fp16=False,  # Disable mixed precision to avoid potential issues
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(output_dir, "final_model"))
    return model

# Generate text with the model
def generate_text(model, input_text, max_length=256):
    # Preprocess the input like we did for training
    processed_input = preprocess_text(input_text, lemmatize=True, remove_stopwords=False)
    
    input_ids = tokenizer.encode("summarize: " + processed_input, return_tensors="pt", max_length=1024, truncation=True)
    
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Validate model
def validate_model(model, val_dataset, num_examples=3):
    # Take a few examples from validation set
    examples = val_dataset.select(range(min(num_examples, len(val_dataset))))
    
    for idx, example in enumerate(examples):
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        # Remove the task prefix
        input_text = input_text.replace("summarize: ", "")
        
        reference_text = tokenizer.decode(example["labels"], skip_special_tokens=True)
        
        generated_text = generate_text(model, input_text)
        
        print(f"\nExample {idx+1}:")
        print(f"Input preview: {input_text[:200]}...")
        print(f"Generated abstract: {generated_text}")
        print(f"Reference abstract: {reference_text}")
        print("-" * 80)

# Main function
if __name__ == "__main__":
    # Use absolute file paths
    project_dir = "C:/Users/vijay/Desktop/research paper summarizer/research-paper-analyser"
    
    # Get a list of all PDFs in the dataset directory
    dataset_dir = os.path.join(project_dir, "training dataset")
    pdf_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {dataset_dir}")
    
    # Limit number of files for initial testing
    max_files = 10  # Start with a small number to test
    pdf_files = pdf_files[:max_files]
    
    output_dir = os.path.join(project_dir, "results")
    
    # Debug information
    print(f"Found {len(pdf_files)} PDF files for training")
    print(f"First few files: {pdf_files[:3]}")
    print(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading and preprocessing data...")
    # Load datasets
    full_dataset = load_and_preprocess_dataset(pdf_files)
    
    print(f"Dataset size after preprocessing: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        raise ValueError("No valid paper abstracts were extracted. Please check your dataset.")
    
    # Split dataset into train and validation
    train_val_dict = full_dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_dict["train"]
    val_dataset = train_val_dict["test"]
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Fine-tune model
    print("Starting fine-tuning...")
    model = fine_tune_model(train_dataset, val_dataset, output_dir)
    
    # Validate model
    print("Validating model...")
    validate_model(model, val_dataset)
    
    print(f"Done! Model saved to {os.path.join(output_dir, 'final_model')}")