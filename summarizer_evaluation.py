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
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

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
            print(f"Processing file {i+1}/{len(file_paths)}: {Path(file_path).name}")
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

# Fine-tune model with increased learning rate
def fine_tune_model(train_dataset, val_dataset, output_dir):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # IMPROVED: Increased learning rate from 5e-5 to 1e-4 for better performance
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,  # Increased from 5e-5
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=10,  # Increased from 8
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU is available
        logging_steps=10,  # Log more frequently
        warmup_steps=100,  # Added warmup steps
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    # Save the final model
    final_model_path = Path(output_dir) / "final_model"
    trainer.save_model(final_model_path)
    
    # Save tokenizer alongside the model for easier evaluation
    tokenizer.save_pretrained(final_model_path)
    
    return model

# Generate text with the model
def generate_text(model, input_text, max_length=256):
    # Preprocess the input like we did for training
    processed_input = preprocess_text(input_text, lemmatize=True, remove_stopwords=False)
    
    input_ids = tokenizer.encode("summarize: " + processed_input, return_tensors="pt", max_length=1024, truncation=True)
    
    # Move to the same device as the model
    device = next(model.parameters()).device
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
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Validate model with improved metrics
def validate_model(model, val_dataset, num_examples=3):
    # Initialize ROUGE scorer
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoother = SmoothingFunction().method1
    
    # Take a few examples from validation set
    examples = val_dataset.select(range(min(num_examples, len(val_dataset))))
    
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    total_bleu = 0
    
    for idx, example in enumerate(examples):
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        # Remove the task prefix
        input_text = input_text.replace("summarize: ", "")
        
        reference_text = tokenizer.decode(example["labels"], skip_special_tokens=True)
        
        generated_text = generate_text(model, input_text)
        
        # Calculate ROUGE scores
        rouge_scores = rouge_scorer_instance.score(reference_text, generated_text)
        
        # Calculate BLEU score
        try:
            reference_tokens = [reference_text.split()]
            candidate_tokens = generated_text.split()
            
            if reference_tokens[0] and candidate_tokens:
                bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoother)
            else:
                bleu = 0.0
        except Exception as e:
            print(f"Error calculating BLEU for example {idx+1}: {e}")
            bleu = 0.0
        
        # Accumulate scores
        total_rouge1 += rouge_scores['rouge1'].fmeasure
        total_rouge2 += rouge_scores['rouge2'].fmeasure
        total_rougeL += rouge_scores['rougeL'].fmeasure
        total_bleu += bleu
        
        print(f"\nExample {idx+1}:")
        print(f"Input preview: {input_text[:500]}...")
        print(f"Generated abstract: {generated_text}")
        print(f"Reference abstract: {reference_text}")
        print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}, ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}, ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}, BLEU: {bleu:.4f}")
        print("-" * 80)
    
    # Calculate averages
    avg_rouge1 = total_rouge1 / num_examples
    avg_rouge2 = total_rouge2 / num_examples
    avg_rougeL = total_rougeL / num_examples
    avg_bleu = total_bleu / num_examples
    
    print(f"\nAverage scores:")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    print(f"BLEU: {avg_bleu:.4f}")

# Save validation results to CSV for later comparison
def save_validation_results(model, val_dataset, output_dir):
    results = []
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoother = SmoothingFunction().method1
    
    for idx, example in enumerate(tqdm(val_dataset, desc="Validating")):
        try:
            input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            input_text = input_text.replace("summarize: ", "")
            
            reference_text = tokenizer.decode(example["labels"], skip_special_tokens=True)
            
            generated_text = generate_text(model, input_text)
            
            # Calculate ROUGE scores
            rouge_scores = rouge_scorer_instance.score(reference_text, generated_text)
            
            # Calculate BLEU score
            try:
                reference_tokens = [reference_text.split()]
                candidate_tokens = generated_text.split()
                
                if reference_tokens[0] and candidate_tokens:
                    bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoother)
                else:
                    bleu = 0.0
            except Exception as e:
                bleu = 0.0
            
            results.append({
                'example_id': idx,
                'reference_text': reference_text,
                'generated_text': generated_text,
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure,
                'bleu': bleu
            })
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
    
    # Save results
    results_path = Path(output_dir) / 'validation_results.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    
    # Calculate and print averages
    avg_scores = {
        'ROUGE-1': results_df['rouge1'].mean(),
        'ROUGE-2': results_df['rouge2'].mean(),
        'ROUGE-L': results_df['rougeL'].mean(),
        'BLEU': results_df['bleu'].mean()
    }
    
    print("\nAverage Validation Scores:")
    for metric, value in avg_scores.items():
        print(f"{metric}: {value:.4f}")
    
    return results_df

# Main function
if __name__ == "__main__":
    # Use Path for more consistent path handling across platforms
    project_dir = input("Enter the path to your project directory (default: current directory): ").strip()
    if not project_dir:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)
    
    # Get a list of all PDFs in the dataset directory
    # Try multiple possible directory names
    possible_dataset_dirs = [
        project_dir / "training_dataset",
        project_dir / "training dataset",
        project_dir / "dataset",
        project_dir / "data"
    ]
    
    dataset_dir = None
    for dir_path in possible_dataset_dirs:
        if dir_path.exists() and dir_path.is_dir():
            dataset_dir = dir_path
            print(f"Found dataset directory: {dataset_dir}")
            break
    
    if not dataset_dir:
        # Ask the user for the correct dataset path
        dataset_dir_input = input("Dataset directory not found. Please enter the full path to your dataset directory: ").strip()
        
        if not dataset_dir_input or not Path(dataset_dir_input).exists():
            dataset_dir = project_dir / "training_dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created dataset directory at {dataset_dir}")
            print("Please add PDF files to this directory and run the script again.")
            exit(1)
        else:
            dataset_dir = Path(dataset_dir_input)
    
    # CORRECTION: Case-insensitive check for PDF extensions
    pdf_files = [file_path for file_path in dataset_dir.glob("*.[pP][dD][fF]")]
    
    if not pdf_files:
        print(f"No PDF files found in {dataset_dir}")
        print("Please add PDF files to this directory and run the script again.")
        exit(1)
    
    # Ask for the number of files to use for training
    while True:
        try:
            max_files_input = input(f"Found {len(pdf_files)} PDF files. How many do you want to use for training? (default: 10): ").strip()
            max_files = int(max_files_input) if max_files_input else 10
            if max_files <= 0:
                print("Please enter a positive number.")
                continue
            if max_files > len(pdf_files):
                max_files = len(pdf_files)
                print(f"Using all {max_files} available files.")
            break
        except ValueError:
            print("Please enter a valid number.")
    
    pdf_files = pdf_files[:max_files]
    
    output_dir = project_dir / "results"
    
    # Debug information
    print(f"Found {len(pdf_files)} PDF files for training")
    print(f"First few files: {[file_path.name for file_path in pdf_files[:3]]}")
    print(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading and preprocessing data...")
    # Load datasets
    full_dataset = load_and_preprocess_dataset(pdf_files)
    
    print(f"Dataset size after preprocessing: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        raise ValueError("No valid paper abstracts were extracted. Please check your dataset.")
    
    # Split dataset into train and validation
    train_val_dict = full_dataset.train_test_split(test_size=0.2)
    train_dataset = train_val_dict["train"]
    val_dataset = train_val_dict["test"]
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Fine-tune model
    print("Starting fine-tuning...")
    model = fine_tune_model(train_dataset, val_dataset, output_dir)
    
    # Validate model
    print("Validating model...")
    validate_model(model, val_dataset, num_examples=3)
    
    # Save comprehensive validation results
    print("Saving validation results...")
    save_validation_results(model, val_dataset, output_dir)
    
    final_model_path = output_dir / 'final_model'
    print(f"Done! Model saved to {final_model_path}")
    
    # Create a CSV file for test data
    test_data_path = project_dir / "test_data.csv"
    if not test_data_path.exists():
        print(f"Creating test data file at {test_data_path}")
        # Extract a few examples from the validation set
        test_examples = val_dataset.select(range(min(5, len(val_dataset))))
        
        test_data = {
            'text': [],
            'abstract': []
        }
        
        for example in test_examples:
            input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            input_text = input_text.replace("summarize: ", "")
            
            reference_text = tokenizer.decode(example["labels"], skip_special_tokens=True)
            
            test_data['text'].append(input_text)
            test_data['abstract'].append(reference_text)
        
        pd.DataFrame(test_data).to_csv(test_data_path, index=False)
        print(f"Created test data file with {len(test_data['text'])} examples")