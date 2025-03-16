import json
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import csv
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


## CREATE SEPARATE FILES

# Function to load JSON data from a file
def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

# Function to create separate lists based on the input and instruction language
def separate_samples(json_data, instruction_lang):
    # Initialize the 3 lists for each language combination with this instruction language
    data = {
        f"{instruction_lang.upper()}_EN": [],
        f"{instruction_lang.upper()}_IT": [],
        f"{instruction_lang.upper()}_PT": []
    }

    # Go through each sample and check the input language
    for sample in json_data:
        for key in sample:
            if key.startswith('input_'):
                input_lang = key.split('_')[1]  # Extract language code from the key
                # Append the sample to the correct list based on the language combination
                data[f"{instruction_lang.upper()}_{input_lang.upper()}"].append(sample)
                break  # Once the correct input language is found, stop looking for other keys

    return data

# Function to process a single JSON file
def process_file(filename):
    # Load the JSON data
    json_data = load_json(filename)
    if not json_data:
        return {}

    # Determine the instruction language based on the filename
    instruction_lang = filename.split('_')[0].lower()  # Convert to lowercase for consistency

    # Separate the samples by language
    separated_data = separate_samples(json_data, instruction_lang)

    return separated_data

# Main function to process the files
def process_all_files(filenames):
    # Initialize a dictionary to hold all lists
    all_separated_data = {
        'IT_EN': [],
        'IT_IT': [],
        'IT_PT': [],
        'EN_EN': [],
        'EN_IT': [],
        'EN_PT': [],
        'PT_EN': [],
        'PT_IT': [],
        'PT_PT': []
    }

    for filename in filenames:
        # Process each file and collect the separated data
        separated_data = process_file(filename)

        # Merge the data into the all_separated_data dictionary
        for key, value in separated_data.items():
            all_separated_data[key].extend(value)

    return all_separated_data


## GENERATE AND TOKENISE PROMPT

# Load the template from the JSON file
def load_templates(template_file):
    try:
        with open(template_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading templates from {template_file}: {e}")
        return []

# Function to select the template based on the instruction language
def select_template(instruction_lang, templates):
    for template in templates:
        if template['lang'].lower() == instruction_lang.lower():  # Case-insensitive comparison
            return template
    return None  # Return None if no matching template is found

# Function to extract language from the filename
def extract_language_from_filename(filename):
    # This regex matches the pattern '{lang}_L_filled_test.json', where {lang} is a two-letter language code (e.g., 'en', 'it', 'pt')
    # Made case-insensitive with (?i)
    match = re.search(r'(?i)([a-z]{2})_L_filled_test\.json', filename)
    if match:
        return match.group(1).lower()  # Extract and return lowercase language code
    else:
        # Also try matching a simpler pattern like 'IT_prova.json'
        simple_match = re.search(r'(?i)^([a-z]{2})_', filename)
        if simple_match:
            return simple_match.group(1).lower()
        else:
            print(f"Warning: Filename {filename} does not match expected pattern. Using default 'en'.")
            return 'en'  # Default to English if pattern doesn't match

def generate_prompt(sample, templates, filename):
    instruction_lang = extract_language_from_filename(filename)
    # Select the template based on the instruction language
    template = select_template(instruction_lang, templates)

    if not template:
        print(f"Warning: No template found for language: {instruction_lang}. Using a basic template.")
        # Fallback template if none is found
        prompt = f"Instruction: {sample['instruction']}\nInput: {sample.get('input_en') or sample.get('input_it') or sample.get('input_pt')}"
    else:
        input_text = sample.get('input_en') or sample.get('input_it') or sample.get('input_pt', '')
        instruction = sample['instruction']

        # Use named parameters in format() to match the {instruction} and {input} placeholders
        prompt = template['prompt_input'].format(
            instruction=instruction,
            input=input_text
        )

    return prompt

def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(sample, templates, tokenizer, filename, cutoff_len):
    # Generate the prompt
    full_prompt = generate_prompt(
        sample=sample,
        templates=templates,
        filename=filename
    )

    # Get the raw instruction and input for reference
    instruction = sample.get('instruction', '')
    input_text = sample.get('input_en') or sample.get('input_it') or sample.get('input_pt', '')
    output = sample.get('output', '')

    # Tokenize the full prompt
    tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len, add_eos_token=True)

    # Store the raw text versions for later reference
    tokenized_full_prompt["raw_instruction"] = instruction
    tokenized_full_prompt["raw_input"] = input_text
    tokenized_full_prompt["raw_output"] = output
    tokenized_full_prompt["raw_prompt"] = full_prompt  # Store the full prompt for generation

    return tokenized_full_prompt


## PROMPT LLAMA
# Function to load model and tokenizer
def load_model_and_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Check if CUDA is available and set the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load the model and move it to the GPU
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)

        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise

# Define stopping criteria class
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids_list):
        self.stop_token_ids_list = stop_token_ids_list  # List of lists of token IDs

    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_token_ids_list:
            # Check if any of the stop sequences appear at the end of the input_ids
            if input_ids.shape[1] >= len(stop_ids):
                # Check if the end of input_ids matches the stop sequence
                if torch.all(input_ids[0, -len(stop_ids):] == torch.tensor(stop_ids, device=input_ids.device)).item():
                    return True
        return False


# Function to generate text from tokenized prompt
def generate_text_from_prompt(tokenized_prompt, tokenizer, model, device, stop_words=None, max_new_tokens=100):
    if stop_words is None:
        stop_words = ["\n\n", "###"]

    # Convert input tensors and move to device
    input_ids = torch.tensor([tokenized_prompt["input_ids"]]).to(device)
    attention_mask = torch.tensor([tokenized_prompt["attention_mask"]]).to(device)

    # Get length of input sequence
    input_length = input_ids.shape[1]

    # Define stopping criteria - encode each stop word as a sequence of token IDs
    stop_token_ids_list = []
    for word in stop_words:
        stop_token_ids = tokenizer(word, return_tensors="pt").input_ids[0].tolist()
        if stop_token_ids:  # Only add if not empty
            stop_token_ids_list.append(stop_token_ids)

    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids_list)])

    try:
        # Generate output from model
        with torch.cuda.amp.autocast():  # Use mixed precision for faster inference
            print("Starting generation...")

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.2,
                repetition_penalty=1.2,
                stopping_criteria=stopping_criteria
            )
            print("Generation complete.")
        # Only decode the newly generated tokens (not the original prompt)
        generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        return generated_text
    except Exception as e:
        print(f"Error in text generation: {e}")
        return ""


# Function to process all tokenized data and generate text
def generate_for_all_samples(tokenized_data, tokenizer, model, device, stop_words=None, max_new_tokens=100, batch_size=1):
    if stop_words is None:
        stop_words = ["\n\n", "Below is", "Di seguito", "Aqui está", "###"]

    generated_data = {
        'IT_EN': [],
        'IT_IT': [],
        'IT_PT': [],
        'EN_EN': [],
        'EN_IT': [],
        'EN_PT': [],
        'PT_EN': [],
        'PT_IT': [],
        'PT_PT': []
    }

    # Loop through all language combinations
    for key, data_list in tokenized_data.items():
        print(f"Generating text for {key}, {len(data_list)} samples")
        for i, tokenized_prompt in enumerate(data_list):
            if i % 10 == 0:  # Progress update
                print(f"  Processing sample {i+1}/{len(data_list)}")

            # Generate the text for the tokenized prompt
            generated_text = generate_text_from_prompt(tokenized_prompt, tokenizer, model, device, stop_words, max_new_tokens)

            # Store the generated text using the raw fields
            generated_data[key].append({
                "instruction": tokenized_prompt["raw_instruction"],
                "input": tokenized_prompt["raw_input"],
                "output": tokenized_prompt["raw_output"],
                "generated_text": generated_text
            })

            # Print a few samples for verification
            if key == "IT_EN" and len(generated_data[key]) <= 5:
                print(f"Sample {len(generated_data[key])}:")
                print(f"Instruction: {tokenized_prompt['raw_instruction']}")
                print(f"Input: {tokenized_prompt['raw_input']}")
                print(f"Output: {tokenized_prompt['raw_output']}")
                print(f"Generated Text: {generated_text}")
                print("-" * 50)

    return generated_data

## COMPUTE METRICS

def assign_label(text):
    """Assign 0 or 1 based on the presence of certain keywords."""
    if text is None:
        return 1  # Default to 1 if text is None

    keywords = ["nessuna", "none", "nenhuma", " no ", " non ", " não "]
    
    text = text.lower()  # Make text lowercase for comparison
    if any(keyword in text for keyword in keywords):
        return 0
    else:
        return 1

def task1_metrics(generated_data):
    """
    Compute Task 1 metrics with language-specific standard deviation for binary idiom detection.
    Uses bootstrap resampling to calculate standard deviations for each language combination.
    """
    # Prepare storage for the results
    metrics_results = {}
    all_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }
    
    # Number of bootstrap iterations
    n_bootstrap = 1000

    # Iterate over the language combinations
    for key, samples in generated_data.items():
        if not samples:  # Skip empty lists
            print(f"Warning: No samples for {key}")
            continue

        generated_labels = []
        output_labels = []

        # Iterate through each sample in the dataset
        for sample in samples:
            # Assign labels to generated_text and output
            generated_label = assign_label(sample.get('generated_text', ''))
            output_label = assign_label(sample.get('output', ''))

            # Append the labels for comparison
            generated_labels.append(generated_label)
            output_labels.append(output_label)

        # Compute the metrics for the comparison between generated_text and output
        try:
            # Calculate base metrics
            acc = accuracy_score(output_labels, generated_labels)
            precision = precision_score(output_labels, generated_labels, zero_division=0)
            recall = recall_score(output_labels, generated_labels, zero_division=0)
            f1 = f1_score(output_labels, generated_labels, zero_division=0)
            
            # Store for overall average calculation
            all_metrics["accuracy"].append(acc)
            all_metrics["precision"].append(precision)
            all_metrics["recall"].append(recall)
            all_metrics["f1"].append(f1)
            
            # Create paired data for bootstrapping
            paired_data = list(zip(output_labels, generated_labels))
            
            # Bootstrap to calculate standard deviation
            bootstrap_accs = []
            bootstrap_precisions = []
            bootstrap_recalls = []
            bootstrap_f1s = []
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                bootstrap_indices = np.random.choice(len(paired_data), len(paired_data), replace=True)
                bootstrap_sample = [paired_data[i] for i in bootstrap_indices]
                
                # Unzip the pairs
                bootstrap_outputs, bootstrap_generated = zip(*bootstrap_sample)
                
                # Calculate metrics for this bootstrap sample
                try:
                    b_acc = accuracy_score(bootstrap_outputs, bootstrap_generated)
                    b_precision = precision_score(bootstrap_outputs, bootstrap_generated, zero_division=0)
                    b_recall = recall_score(bootstrap_outputs, bootstrap_generated, zero_division=0)
                    b_f1 = f1_score(bootstrap_outputs, bootstrap_generated, zero_division=0)
                    
                    bootstrap_accs.append(b_acc)
                    bootstrap_precisions.append(b_precision)
                    bootstrap_recalls.append(b_recall)
                    bootstrap_f1s.append(b_f1)
                except Exception as e:
                    # Handle potential errors in bootstrap calculations
                    print(f"Error in bootstrap calculation: {e}")
                    continue
            
            # Calculate standard deviations for this language combination
            acc_std = np.std(bootstrap_accs) if bootstrap_accs else 0
            precision_std = np.std(bootstrap_precisions) if bootstrap_precisions else 0
            recall_std = np.std(bootstrap_recalls) if bootstrap_recalls else 0
            f1_std = np.std(bootstrap_f1s) if bootstrap_f1s else 0
            
        except Exception as e:
            print(f"Error computing metrics for {key}: {e}")
            acc, precision, recall, f1 = 0, 0, 0, 0
            acc_std, precision_std, recall_std, f1_std = 0, 0, 0, 0

        # Save the metrics in the results dictionary for each dataset
        metrics_results[key] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy_std": acc_std,
            "precision_std": precision_std,
            "recall_std": recall_std,
            "f1_std": f1_std,
            # Store individual predictions for reference
            "individual_predictions": list(zip(output_labels, generated_labels))
        }

    # Calculate means for overall metrics
    means = {
        "accuracy": np.mean(all_metrics["accuracy"]),
        "precision": np.mean(all_metrics["precision"]),
        "recall": np.mean(all_metrics["recall"]),
        "f1": np.mean(all_metrics["f1"])
    }
    
    # Calculate standard deviations across language combinations 
    # (useful for understanding variability between languages)
    across_langs_std_devs = {
        "accuracy": np.std(all_metrics["accuracy"]),
        "precision": np.std(all_metrics["precision"]),
        "recall": np.std(all_metrics["recall"]),
        "f1": np.std(all_metrics["f1"])
    }

    # Save the results to a TSV file
    try:
        with open('task1_metrics_with_bootstrap_std.tsv', 'w', newline='', encoding='utf-8') as tsv_file:
            fieldnames = [
                'language_combination', 'accuracy', 'precision', 'recall', 'f1'
            ]
            writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            # Write each language combination's metrics with its own std dev
            for key, metrics in metrics_results.items():
                row = {
                    'language_combination': key,
                    'accuracy': f"{metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}",
                    'precision': f"{metrics['precision']:.4f} ± {metrics['precision_std']:.4f}",
                    'recall': f"{metrics['recall']:.4f} ± {metrics['recall_std']:.4f}",
                    'f1': f"{metrics['f1']:.4f} ± {metrics['f1_std']:.4f}"
                }
                writer.writerow(row)
            
            # Write the overall means and between-language std devs
            writer.writerow({
                'language_combination': 'OVERALL',
                'accuracy': f"{means['accuracy']:.4f} ± {across_langs_std_devs['accuracy']:.4f}",
                'precision': f"{means['precision']:.4f} ± {across_langs_std_devs['precision']:.4f}",
                'recall': f"{means['recall']:.4f} ± {across_langs_std_devs['recall']:.4f}",
                'f1': f"{means['f1']:.4f} ± {across_langs_std_devs['f1']:.4f}"
            })

        print("Metrics have been saved to task1_metrics_with_bootstrap_std.tsv.")
    except Exception as e:
        print(f"Error saving metrics to file: {e}")
    
    return metrics_results, across_langs_std_devs, means

# Helper function to compute partial character overlap (simple version)
def longest_common_subsequence(str1, str2):
    """
    Find the longest common subsequence between two strings.
    This allows for non-consecutive characters that maintain the same order.
    """
    if not str1 or not str2:  # Check for empty strings
        return 0, ""
    
    # Create a table to store lengths of LCS
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the subsequence
    i, j = m, n
    subseq = []
    
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            subseq.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    # Reverse the subsequence and convert to string
    subsequence = ''.join(reversed(subseq))
    return dp[m][n], subsequence  # Return length and the subsequence itself


# We need to modify compute_overlap to calculate both precision and recall per the formulas
def compute_overlap_with_metrics(gold_span, predicted_span):
    """
    Compute overlap metrics according to the partial text span matching formulas:
    - Precision = |s∩t|/|t| (intersection / length of gold span)
    - Recall = |s∩t|/|s| (intersection / length of predicted span)
    
    Returns precision, recall, overlap_score, and common subsequence
    """
    # Handle None values
    gold_span = gold_span or ""
    predicted_span = predicted_span or ""

    # Clean the texts by stripping unwanted characters (spaces, punctuation)
    gold_clean = ''.join(c.lower() for c in gold_span if c.isalnum())
    predicted_clean = ''.join(c.lower() for c in predicted_span if c.isalnum())

    # Handle empty strings to prevent division by zero
    if not gold_clean or not predicted_clean:
        if not gold_clean and not predicted_clean:
            return 1.0, 1.0, 1.0, ""  # Both empty strings have 100% match
        return 0.0, 0.0, 0.0, ""  # One empty string and one non-empty string have 0% match

    # Compute the length of the longest common subsequence
    lcs_length, common_subsequence = longest_common_subsequence(gold_clean, predicted_clean)

    # Calculate metrics according to the formulas
    precision = lcs_length / len(gold_clean)      # |s∩t|/|t| where t is gold span
    recall = lcs_length / len(predicted_clean)    # |s∩t|/|s| where s is predicted span
    
    # Normalize by the length of the shorter of the two strings for the general overlap score
    overlap_score = lcs_length / min(len(gold_clean), len(predicted_clean))

    return precision, recall, overlap_score, common_subsequence

# Function to compute metrics and save to TSV for Task 2
def task2_metrics(generated_data):
    """
    Compute Task 2 metrics with language-specific standard deviations using bootstrap resampling.
    Uses the partial text span matching formulas:
    - Precision P(S,T) = (1/|S|) × ∑ᵈ∈ᴰ ∑ₛ∈ₛₗ,ₜ∈ₜₗ (|s∩t|/|t|)
    - Recall R(S,T) = (1/|T|) × ∑ᵈ∈ᴰ ∑ₛ∈ₛₗ,ₜ∈ₜₗ (|s∩t|/|s|)
    
    Only samples with label = 1 (idioms present) are included in the evaluation.
    """
    print("Starting Task 2 metrics computation with bootstrap standard deviations...")
    
    # Prepare storage for the results
    metrics_results = {}
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "avg_overlap": []
    }
    
    # Number of bootstrap iterations
    n_bootstrap = 1000

    # Storage for high overlap and low overlap samples
    high_overlap_samples = []
    low_overlap_samples = []
    
    # Store all individual overlap scores for cross-language SD calculation
    all_overlap_scores = []
    all_precision_values = []
    all_recall_values = []

    # Iterate over the language combinations
    for key, samples in generated_data.items():
        if not samples:  # Skip empty lists
            print(f"Warning: No samples for {key}")
            continue

        overlap_scores = []
        individual_precision = []
        individual_recall = []
        samples_with_label_1 = 0  # Count samples with label 1
        individual_data = []  # Store data for bootstrap
        
        # Iterate through each sample in the dataset
        for sample in samples:
            # Check if this is a sample with label = 1 (has an idiom)
            output_label = assign_label(sample.get('output', ''))
            
            if output_label == 1:  # Only process samples with idioms present
                samples_with_label_1 += 1
                
                # Get the text spans
                gold_span = sample.get('output', '').strip()
                predicted_span = sample.get('generated_text', '').strip()
                
                # Skip if either span is empty after cleaning
                if not gold_span or not predicted_span:
                    continue
                
                # Use compute_overlap_with_metrics to get precision, recall, overlap_score
                precision, recall, overlap_score, common_subsequence = compute_overlap_with_metrics(gold_span, predicted_span)
                
                # Store the values
                overlap_scores.append(overlap_score)
                individual_precision.append(precision)
                individual_recall.append(recall)
                
                # Add to global lists for cross-language standard deviation
                all_overlap_scores.append(overlap_score)
                all_precision_values.append(precision)
                all_recall_values.append(recall)
                
                # Store individual data for bootstrap resampling
                individual_data.append({
                    'precision': precision,
                    'recall': recall,
                    'overlap_score': overlap_score
                })

                # Categorize as high or low overlap
                if precision >= 0.5 and recall >= 0.5:
                    high_overlap_samples.append({
                        'language_combination': key,
                        'sample': sample,
                        'overlap_score': overlap_score,
                        'precision': precision,
                        'recall': recall,
                        'common_subsequence': common_subsequence
                    })
                else:
                    low_overlap_samples.append({
                        'language_combination': key,
                        'sample': sample,
                        'overlap_score': overlap_score,
                        'precision': precision,
                        'recall': recall,
                        'common_subsequence': common_subsequence
                    })

        # Skip metrics calculation if no samples had label 1
        if samples_with_label_1 == 0 or not individual_precision or not individual_recall:
            print(f"Warning: No valid samples with label 1 found in {key}")
            metrics_results[key] = {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "avg_overlap": 0,
                "samples_processed": 0,
                "precision_std": 0,
                "recall_std": 0,
                "f1_std": 0,
                "avg_overlap_std": 0
            }
            continue

        # Calculate aggregate metrics following the formulas
        try:
            # Following P(S,T) formula: average precision across all samples
            precision = sum(individual_precision) / len(individual_precision)
            
            # Following R(S,T) formula: average recall across all samples
            recall = sum(individual_recall) / len(individual_recall)
            
            # Calculate F1 score as the harmonic mean of precision and recall
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate average overlap score
            avg_overlap = sum(overlap_scores) / len(overlap_scores)
            
            # Store metrics for calculating cross-language standard deviation
            all_metrics["precision"].append(precision)
            all_metrics["recall"].append(recall)
            all_metrics["f1"].append(f1)
            all_metrics["avg_overlap"].append(avg_overlap)
            
            # Bootstrap to calculate language-specific standard deviations
            bootstrap_precisions = []
            bootstrap_recalls = []
            bootstrap_f1s = []
            bootstrap_overlaps = []
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                bootstrap_indices = np.random.choice(len(individual_data), len(individual_data), replace=True)
                bootstrap_sample = [individual_data[i] for i in bootstrap_indices]
                
                # Calculate metrics for this bootstrap sample
                b_precision = sum(item['precision'] for item in bootstrap_sample) / len(bootstrap_sample)
                b_recall = sum(item['recall'] for item in bootstrap_sample) / len(bootstrap_sample)
                b_f1 = 2 * (b_precision * b_recall) / (b_precision + b_recall) if (b_precision + b_recall) > 0 else 0
                b_overlap = sum(item['overlap_score'] for item in bootstrap_sample) / len(bootstrap_sample)
                
                bootstrap_precisions.append(b_precision)
                bootstrap_recalls.append(b_recall)
                bootstrap_f1s.append(b_f1)
                bootstrap_overlaps.append(b_overlap)
            
            # Calculate standard deviations for this language combination
            precision_std = np.std(bootstrap_precisions)
            recall_std = np.std(bootstrap_recalls)
            f1_std = np.std(bootstrap_f1s)
            overlap_std = np.std(bootstrap_overlaps)
            
        except Exception as e:
            print(f"Error computing metrics for {key}: {e}")
            precision, recall, f1, avg_overlap = 0, 0, 0, 0
            precision_std, recall_std, f1_std, overlap_std = 0, 0, 0, 0

        # Save the metrics in the results dictionary for each dataset
        metrics_results[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_overlap": avg_overlap,
            "samples_processed": samples_with_label_1,
            "precision_std": precision_std,
            "recall_std": recall_std,
            "f1_std": f1_std,
            "avg_overlap_std": overlap_std,
            "individual_precision": individual_precision,
            "individual_recall": individual_recall,
            "individual_overlaps": overlap_scores
        }

    # Calculate means for overall metrics
    means = {
        "precision": np.mean(all_precision_values) if all_precision_values else 0,
        "recall": np.mean(all_recall_values) if all_recall_values else 0,
        "f1": np.mean(all_metrics["f1"]),
        "avg_overlap": np.mean(all_metrics["avg_overlap"]),
        "individual_overlap": np.mean(all_overlap_scores) if all_overlap_scores else 0
    }

    # Calculate cross-language standard deviations
    across_langs_std_devs = {
        "precision": np.std(all_metrics["precision"]),
        "recall": np.std(all_metrics["recall"]),
        "f1": np.std(all_metrics["f1"]),
        "avg_overlap": np.std(all_metrics["avg_overlap"]),
        "individual_overlap": np.std(all_overlap_scores) if all_overlap_scores else 0
    }

    # Save the results to TSV files
    try:
        # Save metrics to main TSV file
        with open('task2_metrics_with_bootstrap_std.tsv', 'w', newline='', encoding='utf-8') as tsv_file:
            fieldnames = ['language_combination', 'precision', 'recall', 'f1', 'avg_overlap', 'samples_processed']
            writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            # Write each language combination's metrics with its own standard deviation
            for key, metrics in metrics_results.items():
                row = {
                    'language_combination': key,
                    'precision': f"{metrics['precision']:.4f} ± {metrics['precision_std']:.4f}",
                    'recall': f"{metrics['recall']:.4f} ± {metrics['recall_std']:.4f}",
                    'f1': f"{metrics['f1']:.4f} ± {metrics['f1_std']:.4f}",
                    'avg_overlap': f"{metrics['avg_overlap']:.4f} ± {metrics['avg_overlap_std']:.4f}",
                    'samples_processed': metrics['samples_processed']
                }
                writer.writerow(row)
            
            # Write the overall means and cross-language standard deviations
            writer.writerow({
                'language_combination': 'OVERALL',
                'precision': f"{means['precision']:.4f} ± {across_langs_std_devs['precision']:.4f}",
                'recall': f"{means['recall']:.4f} ± {across_langs_std_devs['recall']:.4f}",
                'f1': f"{means['f1']:.4f} ± {across_langs_std_devs['f1']:.4f}",
                'avg_overlap': f"{means['avg_overlap']:.4f} ± {across_langs_std_devs['avg_overlap']:.4f}",
                'samples_processed': 'N/A'
            })

        # Save high overlap samples
        with open('high_overlap_samples_with_std.tsv', 'w', newline='', encoding='utf-8') as high_file:
            fieldnames = ['language_combination', 'output', 'generated_text', 'precision', 'recall', 'overlap_score', 'common_subsequence']
            writer = csv.DictWriter(high_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for sample in high_overlap_samples:
                writer.writerow({
                    'language_combination': sample['language_combination'],
                    'output': sample['sample'].get('output', ''),
                    'generated_text': sample['sample'].get('generated_text', ''),
                    'precision': f"{sample['precision']:.4f}",
                    'recall': f"{sample['recall']:.4f}",
                    'overlap_score': f"{sample['overlap_score']:.4f}",
                    'common_subsequence': sample['common_subsequence']
                })

        # Save low overlap samples
        with open('low_overlap_samples_with_std.tsv', 'w', newline='', encoding='utf-8') as low_file:
            fieldnames = ['language_combination', 'output', 'generated_text', 'precision', 'recall', 'overlap_score', 'common_subsequence']
            writer = csv.DictWriter(low_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for sample in low_overlap_samples:
                writer.writerow({
                    'language_combination': sample['language_combination'],
                    'output': sample['sample'].get('output', ''),
                    'generated_text': sample['sample'].get('generated_text', ''),
                    'precision': f"{sample['precision']:.4f}",
                    'recall': f"{sample['recall']:.4f}",
                    'overlap_score': f"{sample['overlap_score']:.4f}",
                    'common_subsequence': sample['common_subsequence']
                })

        print("Task 2 metrics with bootstrap standard deviations and sample data have been saved.")
    except Exception as e:
        print(f"Error saving metrics to file: {e}")
    
    return metrics_results, across_langs_std_devs, means
    

# Function to collect wrong predictions for Task 1 and Task 2
def collect_wrong_predictions(generated_data):
    """Collect and save wrong predictions for Task 1 and Task 2."""
    wrong_predictions = {}  # Use a dictionary to avoid duplicates
    
    # Counters for each type of error
    task1_errors = 0
    task2_errors = 0
    both_task_errors = 0

    # Iterate over all language combinations
    for key, samples in generated_data.items():
        for idx, sample in enumerate(samples):
            sample_id = f"{key}_{idx}"  # Create a unique ID for each sample
            wrong_for_tasks = []

            # Task 1 - Compare generated label and output label
            generated_label = assign_label(sample.get('generated_text', ''))
            output_label = assign_label(sample.get('output', ''))

            # If the generated label does not match the output label, it's a wrong prediction
            if generated_label != output_label:
                wrong_for_tasks.append('task 1')
                task1_errors += 1

            # Task 2 - Only evaluate overlap for samples with label = 1
            if output_label == 1:
                # Compute character overlap score
                precision, recall, overlap_score, common_subsequence = compute_overlap_with_metrics(
                sample.get('output', ''), 
                sample.get('generated_text', '')
)

                # If the overlap score is less than 0.1, it's a wrong prediction for Task 2
                if overlap_score < 0.1:
                    wrong_for_tasks.append('task 2')
                    task2_errors += 1
                
                # Add overlap info to the sample data
                overlap_info = {
                    'overlap_score': overlap_score,
                    'common_subsequence': common_subsequence if overlap_score > 0 else ''
                }
            else:
                # Not applicable for task 2
                overlap_info = {
                    'overlap_score': 'N/A',
                    'common_subsequence': 'N/A'
                }

            # Add to wrong_predictions if it's wrong for any task
            if wrong_for_tasks:
                if len(wrong_for_tasks) > 1:
                    both_task_errors += 1
                    
                wrong_predictions[sample_id] = {
                    'language_combination': key,
                    'instruction': sample.get('instruction', ''),
                    'input': sample.get('input', ''),
                    'output': sample.get('output', ''),
                    'generated_text': sample.get('generated_text', ''),
                    'wrong_for_task': ', '.join(wrong_for_tasks),
                    **overlap_info
                }

    # Save wrong predictions to a TSV file
    try:
        with open('wrong_predictions_with_stats.tsv', 'w', newline='', encoding='utf-8') as tsv_file:
            fieldnames = [
                'language_combination', 'instruction', 'input', 'output', 'generated_text',
                'wrong_for_task', 'overlap_score', 'common_subsequence'
            ]
            writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            # Write each wrong prediction as a row in the file
            for prediction in wrong_predictions.values():
                writer.writerow(prediction)

            # Add statistics at the end of the file
            total_samples = sum(len(samples) for samples in generated_data.values())
            error_stats = [
                {'language_combination': '===STATISTICS==='},
                {'language_combination': f'Total samples: {total_samples}'},
                {'language_combination': f'Task 1 errors: {task1_errors} ({task1_errors/total_samples*100:.2f}%)'},
                {'language_combination': f'Task 2 errors: {task2_errors}'},
                {'language_combination': f'Errors in both tasks: {both_task_errors}'}
            ]
            
            for stat in error_stats:
                writer.writerow(stat)

        print(f"Wrong predictions have been saved to wrong_predictions_with_stats.tsv. Total: {len(wrong_predictions)}")
    except Exception as e:
        print(f"Error saving wrong predictions to file: {e}")
    
    return wrong_predictions

# Function to calculate bootstrap confidence intervals for metrics
def bootstrap_confidence_intervals(metrics, individual_data, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for metrics.
    
    Parameters:
    metrics (dict): Dictionary of metrics
    individual_data (list): List of individual data points
    n_bootstrap (int): Number of bootstrap samples
    confidence_level (float): Confidence level for interval
    
    Returns:
    dict: Dictionary with confidence intervals for each metric
    """
    bootstrap_metrics = {metric: [] for metric in metrics.keys()}
    
    # Generate bootstrap samples
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(individual_data, size=len(individual_data), replace=True)
        
        # Calculate metrics for bootstrap sample
        for metric, func in metrics.items():
            bootstrap_metrics[metric].append(func(bootstrap_sample))
    
    # Calculate confidence intervals
    alpha = (1 - confidence_level) / 2
    confidence_intervals = {}
    
    for metric, values in bootstrap_metrics.items():
        lower = np.percentile(values, alpha * 100)
        upper = np.percentile(values, (1 - alpha) * 100)
        confidence_intervals[metric] = (lower, upper)
    
    return confidence_intervals

def main():
    # Model configuration
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    # Files to process
    #files = ['IT_L_filled_test.json', 'EN_L_filled_test.json', 'PT_L_filled_test.json']
    files = ['IT_prova.json']

    try:
        # Verify CUDA is available
        if torch.cuda.is_available():
            print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU instead.")

        # Load templates
        templates = load_templates('templates.json')
        if not templates:
            raise ValueError("Failed to load templates")

        # Load model and tokenizer
        print(f"Loading model and tokenizer: {model_name}")
        model, tokenizer, device = load_model_and_tokenizer(model_name)

        # Process all files
        print("Processing input files...")
        separated_data = process_all_files(files)

        # Create filename mapping for proper file reference
        lang_to_file = {
            'en': 'EN_L_filled_test.json',
            'it': 'IT_L_filled_test.json',
            'pt': 'PT_L_filled_test.json'
        }

        # Generate tokenized data for all samples
        print("Tokenizing data...")
        tokenized_data = {}
        for key, samples in separated_data.items():
            instruction_lang = key.split('_')[0].lower()  # Extract the instruction language from the key
            tokenized_data[key] = []
            for sample in samples:
                # Use the correct filename based on the instruction language
                file_for_sample = lang_to_file.get(instruction_lang.lower(), files[0])
                tokenized_prompt = generate_and_tokenize_prompt(
                    sample=sample,
                    templates=templates,
                    tokenizer=tokenizer,
                    filename=file_for_sample,
                    cutoff_len=512
                )
                tokenized_data[key].append(tokenized_prompt)

        # Generate text for all samples
        print("Generating text for all samples...")
        # Set torch.no_grad() for inference to save memory
        with torch.no_grad():
            generated_data = generate_for_all_samples(tokenized_data, tokenizer, model, device)

        # Compute and save metrics with standard deviation
        print("Computing metrics for Task 1 with standard deviation...")
        task1_results, task1_std_devs, task1_means = task1_metrics(generated_data)

        print("Computing metrics for Task 2 with standard deviation...")
        task2_results, task2_std_devs, task2_means = task2_metrics(generated_data)

        # Collect and save wrong predictions with statistics
        print("Collecting wrong predictions with statistics...")
        collect_wrong_predictions(generated_data)

        print("Evaluation completed successfully!")

        # Print overall metrics summary with standard deviation
        print("\nOVERALL METRICS SUMMARY WITH STANDARD DEVIATION")
        print("=" * 80)
        print(f"Task 1 - Accuracy: {task1_means['accuracy']:.4f} ± {task1_std_devs['accuracy']:.4f}")
        print(f"Task 1 - Precision: {task1_means['precision']:.4f} ± {task1_std_devs['precision']:.4f}")
        print(f"Task 1 - Recall: {task1_means['recall']:.4f} ± {task1_std_devs['recall']:.4f}")
        print(f"Task 1 - F1: {task1_means['f1']:.4f} ± {task1_std_devs['f1']:.4f}")
        print("-" * 80)
        print(f"Task 2 - Accuracy: {task2_means['accuracy']:.4f} ± {task2_std_devs['accuracy']:.4f}")
        print(f"Task 2 - Precision: {task2_means['precision']:.4f} ± {task2_std_devs['precision']:.4f}")
        print(f"Task 2 - Recall: {task2_means['recall']:.4f} ± {task2_std_devs['recall']:.4f}")
        print(f"Task 2 - F1: {task2_means['f1']:.4f} ± {task2_std_devs['f1']:.4f}")
        print(f"Task 2 - Average overlap: {task2_means['avg_overlap']:.4f} ± {task2_std_devs['avg_overlap']:.4f}")
        print(f"Task 2 - Individual sample overlap: {task2_means['individual_overlap']:.4f} ± {task2_std_devs['individual_overlap']:.4f}")

    except Exception as e:
        print(f"An error occurred during execution: {e}")

    # Example data
    example_data = [
        {
            "instruction": "An analysis reveals the presence of the following idiomatic expressions",
            "input": "Hey, I got a new phone number. Call me at 555-123-4567",
            "output": "None. ",
            "generated_text": "There is no danger present in this message. The message appears to be a straightforward notification about a change in phone number."
        },
        {
            "instruction": "Can you spot idiomatic expressions in the following sentence?",
            "input": "He kicked the bucket too soon.",
            "output": "kicked the bucket",
            "generated_text": "kicked the"
        }
    ]

    # Demonstrate Task1 and Task2 metrics computation with examples
    print("=" * 80)
    print("METRICS COMPUTATION EXAMPLES")
    print("=" * 80)

    print("\nTASK 1: BINARY CLASSIFICATION (DANGER DETECTION)\n")
    print("This task evaluates whether the model correctly detects if there's danger or not.\n")

    for idx, sample in enumerate(example_data, 1):
        print(f"Example {idx}:")
        print(f"Input: '{sample['input']}'")
        print(f"Expected output: '{sample['output']}'")
        print(f"Generated text: '{sample['generated_text']}'")
        
        # Task 1: Binary classification
        output_label = assign_label(sample['output'])
        generated_label = assign_label(sample['generated_text'])
        
        print(f"\nTask 1 computation:")
        print(f"- Expected label: {output_label} ({'danger' if output_label==1 else 'no danger'})")
        print(f"- Generated label: {generated_label} ({'danger' if generated_label==1 else 'no danger'})")
        print(f"- Correct classification: {output_label == generated_label}")
        
        # Task 2: Overlap score
        precision, recall, overlap_score, common_substring = compute_overlap_with_metrics(sample['output'], sample['generated_text'])
        binary_overlap = 1 if overlap_score >= 0.1 else 0
        
        print(f"\nTask 2 computation:")
        print(f"- Cleaned expected output: '{''.join(c for c in sample['output'] if c.isalnum())}'")
        print(f"- Cleaned generated text: '{''.join(c for c in sample['generated_text'] if c.isalnum())}'")
        print(f"- Longest common substring length: {len(common_substring)}")
        if common_substring:
            print(f"- Longest common substring: '{common_substring}'")
        print(f"- Overlap score: {overlap_score:.4f}")
        print(f"- Binary overlap (≥0.1): {binary_overlap}")
        print(f"- High overlap (≥0.5): {'Yes' if overlap_score >= 0.5 else 'No'}")
        
        print("\n" + "-" * 80 + "\n")

    # Calculate overall metrics
    output_labels = [assign_label(sample['output']) for sample in example_data]
    generated_labels = [assign_label(sample['generated_text']) for sample in example_data]

    # Task 1 metrics
    correct_classifications = sum(1 for o, g in zip(output_labels, generated_labels) if o == g)
    task1_accuracy = correct_classifications / len(example_data)

    # Task 2 metrics
    overlap_scores = [compute_overlap_with_metrics(sample['output'], sample['generated_text'])[0] for sample in example_data]
    binary_overlaps = [1 if score >= 0.1 else 0 for score in overlap_scores]
    high_overlaps = [1 if score >= 0.5 else 0 for score in overlap_scores]
    task2_accuracy = sum(binary_overlaps) / len(binary_overlaps)

    print("OVERALL METRICS SUMMARY")
    print("=" * 80)
    print(f"Task 1 accuracy: {task1_accuracy:.4f}")
    print(f"Task 2 - Meaningful overlap ratio (≥0.1): {task2_accuracy:.4f}")
    print(f"Task 2 - High overlap ratio (≥0.5): {sum(high_overlaps) / len(high_overlaps):.4f}")
    print(f"Task 2 - Average overlap score: {sum(overlap_scores) / len(overlap_scores):.4f}")



if __name__ == "__main__":
    main()
