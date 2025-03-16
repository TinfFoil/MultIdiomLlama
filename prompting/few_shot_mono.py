import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os
import re
import numpy as np

## CREATE SEPARATE FILES

def load_json(filename):
    """Load JSON data from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

def load_templates(template_file):
    """Load prompt templates from a JSON file."""
    try:
        with open(template_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading templates from {template_file}: {e}")
        return []

def load_demonstrations(demo_file):
    """Load demonstrations from a JSON file."""
    try:
        with open(demo_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading demonstrations from {demo_file}: {e}")
        return []

def extract_language_from_filename(filename):
    """Extract language code from the filename."""
    # Extract language code from filename pattern like 'IT.json', 'EN_test.json', etc.
    match = re.search(r'(?i)([a-z]{2})(_|\.|$)', filename)
    if match:
        return match.group(1).lower()  # Return lowercase language code
    else:
        print(f"Warning: Could not extract language from filename {filename}. Defaulting to 'en'.")
        return 'en'  # Default to English if pattern doesn't match

def select_template(instruction_lang, templates):
    """Select template based on the instruction language."""
    for template in templates:
        if template['lang'].lower() == instruction_lang.lower():
            return template
    print(f"Warning: No template found for language: {instruction_lang}. Using default.")
    # Return a default template if none matches
    return {
        "lang": "en",
        "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_demo": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}\n\n"
    }

def select_demonstrations(instruction_lang, demonstrations):
    """
    Select demonstrations matching the instruction language and input language.
    
    Args:
        instruction_lang (str): Language of the instruction (e.g., 'it', 'en', 'pt')
        demonstrations (list): List of demonstration examples
    
    Returns:
        list: Matching demonstrations
    """
    if not demonstrations:
        print("Warning: No demonstrations provided.")
        return []

    matching_demos = []
    
    # Language-specific markers and instruction phrases
    lang_markers = {
        'it': ['frase', 'espressioni', 'costruzioni', 'rilevare', 'individuare'],
        'en': ['sentence', 'idiomatic', 'expressions', 'identify', 'spot'],
        'pt': ['frase', 'expressões', 'construções', 'identificar']
    }
    
    # Look for demonstrations with matching input language
    for demo in demonstrations:
        try:
            # Check if the demonstration has a language-specific input field
            input_key = f"input_{instruction_lang}"
            
            # If the specific input language field exists
            if input_key in demo:
                # Get the instruction as a string
                instruction = str(demo.get('instruction', '')).lower()
                
                # Check for language markers in the instruction
                lang_match = any(
                    marker in instruction 
                    for marker in lang_markers.get(instruction_lang, [])
                )
                
                # If there's a language match, add the demonstration
                if lang_match:
                    matching_demos.append(demo)
        
        except Exception as e:
            print(f"Error processing demonstration: {e}")
            continue
    
    # If no language-specific matches, fall back to all demonstrations with the right input key
    if not matching_demos:
        print(f"Warning: No demonstrations found for language {instruction_lang}. Using alternative approach.")
        
        input_key = f"input_{instruction_lang}"
        matching_demos = [demo for demo in demonstrations if input_key in demo]
    
    # Fallback to any available demonstrations
    if not matching_demos:
        print(f"Warning: No demonstrations found. Using any available demonstrations.")
        matching_demos = demonstrations[:min(3, len(demonstrations))]
    
    # Final sanity check and limit to 3 demonstrations
    matching_demos = matching_demos[:3]
    
    if not matching_demos:
        print("Critical warning: No demonstrations could be selected.")
    
    return matching_demos

def process_file(filename):
    """Process a single JSON file and organize samples by language."""
    json_data = load_json(filename)
    if not json_data:
        return {}
    
    # Determine the instruction language from the filename
    instruction_lang = extract_language_from_filename(filename)
    
    # Separate data by input language (EN, IT, PT)
    separated_data = {
        f"{instruction_lang.upper()}_EN": [],
        f"{instruction_lang.upper()}_IT": [],
        f"{instruction_lang.upper()}_PT": []
    }
    
    # Process each sample
    for sample in json_data:
        # Determine the input language from the keys
        input_lang = None
        for key in sample:
            if key.startswith('input_'):
                input_lang = key.split('_')[1].upper()
                break
        
        # If input language isn't specified in a separate key, try to determine from 'input'
        if not input_lang and 'input' in sample:
            # This is a simplified approach - you might want a more sophisticated language detection
            input_text = sample['input']
            
            # Apply basic heuristics to guess the language
            if any(word in input_text.lower() for word in ['the', 'is', 'and', 'of']):
                input_lang = 'EN'
            elif any(word in input_text.lower() for word in ['il', 'la', 'di', 'e', 'sono']):
                input_lang = 'IT'
            elif any(word in input_text.lower() for word in ['o', 'a', 'de', 'da', 'em']):
                input_lang = 'PT'
            else:
                # Default to same as instruction language if undetermined
                input_lang = instruction_lang.upper()
        
        # Add the sample to the appropriate language combination
        if input_lang:
            key = f"{instruction_lang.upper()}_{input_lang}"
            if key in separated_data:
                separated_data[key].append(sample)
            else:
                # If the key doesn't exist, create it
                separated_data[key] = [sample]
    
    return separated_data

def process_all_files(filenames):
    """Process multiple files and combine their data."""
    all_separated_data = {}
    
    for filename in filenames:
        # Process each file
        separated_data = process_file(filename)
        
        # Add data to the combined dictionary
        for key, samples in separated_data.items():
            if key not in all_separated_data:
                all_separated_data[key] = []
            all_separated_data[key].extend(samples)
    
    return all_separated_data

## GENERATE PROMPTS

def create_demonstration_prompts(demonstrations, template):
    """Create prompts for demonstrations using the template."""
    demo_prompts = []
    
    # Get the template format - for demonstrations we use the same format as for main input
    demo_template = template.get("prompt_input", "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
    
    # Create a prompt for each demonstration
    for demo in demonstrations:
        instruction = demo.get("instruction", "")
        
        # Get the input from the appropriate field based on language
        input_text = demo.get("input", "")
        if not input_text:
            # Try language-specific input fields
            for lang in ["en", "it", "pt"]:
                input_key = f"input_{lang}"
                if input_key in demo:
                    input_text = demo[input_key]
                    break
        
        output = demo.get("output", "")
        
        # Format the demonstration prompt
        # Append the output to the base template
        demo_prompt = demo_template.format(
            instruction=instruction,
            input=input_text
        )
        
        # Add the response/output
        demo_prompt += f"{output}\n\n"
        
        demo_prompts.append(demo_prompt)
    
    return "".join(demo_prompts)

def generate_full_prompt(sample, templates, demonstrations, filename):
    """Generate a full prompt with demonstrations + main sample."""
    # Extract the instruction language from the filename
    instruction_lang = extract_language_from_filename(filename)
    
    # Select the template based on the instruction language
    template = select_template(instruction_lang, templates)
    
    # Select relevant demonstrations for this language
    relevant_demos = select_demonstrations(instruction_lang, demonstrations)
    
    # Create demonstration prompts
    demo_prompts = create_demonstration_prompts(relevant_demos, template)
    
    # Get instruction and input from the sample
    instruction = sample.get('instruction', '')
    
    # Try to get input from language-specific fields, or fall back to the generic input field
    input_text = ""
    for lang in ["en", "it", "pt"]:
        input_key = f"input_{lang}"
        if input_key in sample:
            input_text = sample[input_key]
            break
    
    # If no language-specific input found, try generic input field
    if not input_text and 'input' in sample:
        input_text = sample['input']
    
    # Generate the main prompt
    prompt_template = template.get("prompt_input", "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
    main_prompt = prompt_template.format(instruction=instruction, input=input_text)
    
    # Combine demonstrations and main prompt
    full_prompt = demo_prompts + main_prompt
    
    return full_prompt

def tokenize_prompt(prompt, tokenizer, cutoff_len, add_eos_token=True):
    """Tokenize a prompt for the model."""
    # Make sure we're using the correct return_tensors format
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors="pt"  # This returns PyTorch tensors
    )
    
    # Add EOS token if needed and if there's room
    if add_eos_token and tokenized.input_ids.shape[1] < cutoff_len:
        eos_id = tokenizer.eos_token_id
        # Check if the last token is already the EOS token
        if tokenized.input_ids[0, -1].item() != eos_id:
            # Add EOS token
            new_input_ids = torch.cat([
                tokenized.input_ids, 
                torch.tensor([[eos_id]], device=tokenized.input_ids.device)
            ], dim=1)
            new_attention_mask = torch.cat([
                tokenized.attention_mask,
                torch.tensor([[1]], device=tokenized.attention_mask.device)
            ], dim=1)
            
            tokenized.input_ids = new_input_ids
            tokenized.attention_mask = new_attention_mask
    
    return tokenized


## PROMPT LLAMA

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer with proper device placement."""
    try:
        # Check CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def generate_text(prompt, tokenizer, model, device, stop_words=None, max_new_tokens=100):
    """Generate text from a given prompt."""
    if stop_words is None:
        stop_words = ["\n\n", "###"]
    
    try:
        # Tokenize the prompt directly here (don't use the separate function to ensure consistency)
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors="pt"
        ).to(device)
        
        # Record the input length to extract only the new tokens later
        input_length = inputs.input_ids.shape[1]
        
        # Set up stopping criteria
        stop_token_ids = []
        for word in stop_words:
            # Get the token IDs for each stop word/phrase
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            if word_tokens:
                stop_token_ids.append(word_tokens[0])
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.2,
                repetition_penalty=1.2,
                stopping_criteria=stopping_criteria
            )
        
        # Extract only the newly generated text
        generated_text = tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return generated_text.strip()
    except Exception as e:
        print(f"Error in text generation: {e}")
        import traceback
        traceback.print_exc()
        return ""

def generate_for_all_samples(all_data, templates, demonstrations, tokenizer, model, device):
    """Generate text for all samples in the dataset."""
    generated_data = {}
    
    # Process each language combination
    for key, samples in all_data.items():
        if not samples:
            continue
        
        print(f"Generating responses for {key} ({len(samples)} samples)...")
        generated_data[key] = []
        
        # Extract instruction language from the key
        instruction_lang = key.split('_')[0].lower()
        filename = f"{instruction_lang}.json"  # Create filename for template selection
        
        # Process each sample
        for i, sample in enumerate(samples):
            try:
                # Generate the full prompt with demonstrations
                full_prompt = generate_full_prompt(
                    sample=sample,
                    templates=templates,
                    demonstrations=demonstrations,
                    filename=filename
                )
                
                # Generate text from the prompt
                generated_text = generate_text(
                    prompt=full_prompt,
                    tokenizer=tokenizer,
                    model=model,
                    device=device
                )
                
                # Get the actual input and output for reference
                input_text = ""
                for lang in ["en", "it", "pt"]:
                    input_key = f"input_{lang}"
                    if input_key in sample:
                        input_text = sample[input_key]
                        break
                if not input_text and 'input' in sample:
                    input_text = sample['input']
                
                # Store the results
                generated_data[key].append({
                    "instruction": sample.get("instruction", ""),
                    "input": input_text,
                    "output": sample.get("output", ""),
                    "generated_text": generated_text or "ERROR: Failed to generate text"  # Provide fallback
                })
                
                # Print progress for long runs
                if (i+1) % 10 == 0 or i == len(samples) - 1:
                    print(f"Processed {i+1}/{len(samples)} samples for {key}")
            
            except Exception as e:
                print(f"Error processing sample {i} in {key}: {e}")
                # Add the sample with error information
                generated_data[key].append({
                    "instruction": sample.get("instruction", ""),
                    "input": sample.get("input", ""),
                    "output": sample.get("output", ""),
                    "generated_text": f"ERROR: {str(e)}"
                })
                continue
    
    return generated_data

# COMPUTE METRICS

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

def clean_text_for_comparison(text):
    """Standardize text for comparison by removing non-alphanumeric characters and lowercasing."""
    if not text:
        return ""
    return ''.join(c.lower() for c in text if c.isalnum())

def compute_overlap_with_metrics(gold_span, predicted_span):
    """Compute overlap metrics between two text spans."""
    # Handle None values
    gold_span = gold_span or ""
    predicted_span = predicted_span or ""

    # Clean the texts using the utility function
    gold_clean = clean_text_for_comparison(gold_span)
    predicted_clean = clean_text_for_comparison(predicted_span)

    # Handle empty strings
    if not gold_clean or not predicted_clean:
        if not gold_clean and not predicted_clean:
            return 1.0, 1.0, 1.0, ""  # Both empty strings have 100% match
        return 0.0, 0.0, 0.0, ""  # One empty string and one non-empty string have 0% match

    # Compute LCS
    lcs_length, common_subsequence = longest_common_subsequence(gold_clean, predicted_clean)

    # Calculate metrics
    precision = lcs_length / len(gold_clean)
    recall = lcs_length / len(predicted_clean)
    overlap_score = lcs_length / min(len(gold_clean), len(predicted_clean))

    return precision, recall, overlap_score, common_subsequence

def bootstrap_std_dev(samples, calculate_metric_fn, bootstrap_samples=1000):
    """
    Calculate standard deviation using bootstrapping.
    
    Args:
        samples: List of samples to bootstrap from
        calculate_metric_fn: Function that calculates the metric from a bootstrapped sample
        bootstrap_samples: Number of bootstrap iterations
    
    Returns:
        float: Standard deviation of the bootstrapped metric
    """
    if len(samples) < 5:
        return None  # Not enough samples for reliable bootstrapping
        
    bootstrap_results = []
    
    for _ in range(bootstrap_samples):
        # Sample with replacement
        indices = np.random.randint(0, len(samples), len(samples))
        bootstrap_sample = [samples[i] for i in indices]
        
        # Calculate metric for this bootstrap sample
        metric_value = calculate_metric_fn(bootstrap_sample)
        bootstrap_results.append(metric_value)
    
    # Return the standard deviation
    return np.std(bootstrap_results)

def save_metrics_to_tsv(filename, metrics, fieldnames, global_std_devs, means):
    """
    Save metrics to a TSV file with consistent formatting.
    
    Args:
        filename (str): The filename to save to
        metrics (dict): The metrics data for each language combination
        fieldnames (list): Column headers for the TSV
        global_std_devs (dict): Global standard deviations
        means (dict): Overall means
    """
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as tsv_file:
            writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            # Write each language combination's metrics
            for key, metric_data in metrics.items():
                # Use language-specific standard deviations if available, otherwise global
                std_devs = metric_data.get("std_devs") or global_std_devs
                
                row = {'language_combination': key}
                # Add each metric with its standard deviation
                for field in fieldnames:
                    if field == 'language_combination' or field == 'samples_processed':
                        continue
                    if field in metric_data and field in std_devs:
                        row[field] = f"{metric_data[field]:.4f} ± {std_devs[field]:.4f}"
                
                # Add any non-metric fields
                if 'samples_processed' in fieldnames and 'samples_processed' in metric_data:
                    row['samples_processed'] = metric_data['samples_processed']
                
                writer.writerow(row)
            
            # Write the overall means and standard deviations
            overall_row = {'language_combination': 'OVERALL'}
            for field in fieldnames:
                if field == 'language_combination' or field == 'samples_processed':
                    continue
                if field in means and field in global_std_devs:
                    overall_row[field] = f"{means[field]:.4f} ± {global_std_devs[field]:.4f}"
            
            if 'samples_processed' in fieldnames:
                overall_row['samples_processed'] = 'N/A'
                
            writer.writerow(overall_row)

        print(f"Metrics have been saved to {filename}")
    except Exception as e:
        print(f"Error saving metrics to file: {e}")

def task1_metrics(generated_data):
    """Compute Task 1 metrics with standard deviation for binary idiom detection."""
    # Prepare storage for the results
    metrics_results = {}
    all_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

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

        # Compute the metrics
        try:
            acc = accuracy_score(output_labels, generated_labels)
            precision = precision_score(output_labels, generated_labels, zero_division=0)
            recall = recall_score(output_labels, generated_labels, zero_division=0)
            f1 = f1_score(output_labels, generated_labels, zero_division=0)
            
            # Store metrics for global calculations
            all_metrics["accuracy"].append(acc)
            all_metrics["precision"].append(precision)
            all_metrics["recall"].append(recall)
            all_metrics["f1"].append(f1)
        except Exception as e:
            print(f"Error computing metrics for {key}: {e}")
            acc, precision, recall, f1 = 0, 0, 0, 0

        # Calculate language-specific standard deviation using the utility function
        paired_data = list(zip(output_labels, generated_labels))
        
        # Define metric calculation functions for bootstrapping
        def calc_acc(data): 
            true_labels, pred_labels = zip(*data)
            return accuracy_score(true_labels, pred_labels)
            
        def calc_precision(data): 
            true_labels, pred_labels = zip(*data)
            return precision_score(true_labels, pred_labels, zero_division=0)
            
        def calc_recall(data): 
            true_labels, pred_labels = zip(*data)
            return recall_score(true_labels, pred_labels, zero_division=0)
            
        def calc_f1(data): 
            true_labels, pred_labels = zip(*data)
            return f1_score(true_labels, pred_labels, zero_division=0)
        
        # Get standard deviations using the bootstrap utility
        lang_std_devs = {
            "accuracy": bootstrap_std_dev(paired_data, calc_acc),
            "precision": bootstrap_std_dev(paired_data, calc_precision),
            "recall": bootstrap_std_dev(paired_data, calc_recall),
            "f1": bootstrap_std_dev(paired_data, calc_f1)
        }

        # Save the metrics in the results dictionary
        metrics_results[key] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "std_devs": lang_std_devs,
            "individual_predictions": list(zip(output_labels, generated_labels))
        }

    # Calculate global standard deviations
    global_std_devs = {
        "accuracy": np.std(all_metrics["accuracy"]),
        "precision": np.std(all_metrics["precision"]),
        "recall": np.std(all_metrics["recall"]),
        "f1": np.std(all_metrics["f1"])
    }
    
    # Calculate means for overall metrics
    means = {
        "accuracy": np.mean(all_metrics["accuracy"]),
        "precision": np.mean(all_metrics["precision"]),
        "recall": np.mean(all_metrics["recall"]),
        "f1": np.mean(all_metrics["f1"])
    }

    # Use the shared save_metrics_to_tsv function instead of duplicating code
    save_metrics_to_tsv(
        'task1_metrics_with_std.tsv',
        metrics_results,
        ['language_combination', 'accuracy', 'precision', 'recall', 'f1'],
        global_std_devs,
        means
    )
    
    return metrics_results, global_std_devs, means

def task2_metrics(generated_data):
    """
    Compute and save Task 2 metrics with language-specific standard deviations.
    Only samples with label = 1 (idioms present) are included in the evaluation.
    """
    print("Starting Task 2 metrics computation...")
    # Prepare storage for the results
    metrics_results = {}
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "avg_overlap": []
    }
    
    # Store all individual overlap scores for global SD calculation
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
        
        # Store all samples with idioms for bootstrapping
        idiom_samples = []
        
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
                precision, recall, overlap_score, _ = compute_overlap_with_metrics(gold_span, predicted_span)
                
                # Store sample data for bootstrapping
                idiom_samples.append({
                    "gold": gold_span,
                    "predicted": predicted_span,
                    "precision": precision,
                    "recall": recall,
                    "overlap": overlap_score
                })
                
                # Store the values
                overlap_scores.append(overlap_score)
                individual_precision.append(precision)
                individual_recall.append(recall)
                
                # Add to global lists for standard deviation calculation
                all_overlap_scores.append(overlap_score)
                all_precision_values.append(precision)
                all_recall_values.append(recall)

        # Skip metrics calculation if no samples had label 1
        if samples_with_label_1 == 0 or not individual_precision or not individual_recall:
            print(f"Warning: No valid samples with label 1 found in {key}")
            metrics_results[key] = {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "avg_overlap": 0,
                "samples_processed": 0,
                "std_devs": None
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
            
            # Store metrics for calculating standard deviation
            all_metrics["precision"].append(precision)
            all_metrics["recall"].append(recall)
            all_metrics["f1"].append(f1)
            all_metrics["avg_overlap"].append(avg_overlap)
        except Exception as e:
            print(f"Error computing metrics for {key}: {e}")
            precision, recall, f1, avg_overlap = 0, 0, 0, 0
        
        # Calculate language-specific standard deviations using the utility function
        if len(idiom_samples) >= 5:
            # Define metric calculation functions for bootstrapping
            def calc_precision(samples): 
                return sum(s["precision"] for s in samples) / len(samples)
                
            def calc_recall(samples): 
                return sum(s["recall"] for s in samples) / len(samples)
                
            def calc_f1(samples):
                p = sum(s["precision"] for s in samples) / len(samples)
                r = sum(s["recall"] for s in samples) / len(samples)
                return 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                
            def calc_overlap(samples):
                return sum(s["overlap"] for s in samples) / len(samples)
            
            lang_std_devs = {
                "precision": bootstrap_std_dev(idiom_samples, calc_precision),
                "recall": bootstrap_std_dev(idiom_samples, calc_recall),
                "f1": bootstrap_std_dev(idiom_samples, calc_f1),
                "avg_overlap": bootstrap_std_dev(idiom_samples, calc_overlap)
            }
        else:
            print(f"Not enough samples in {key} for language-specific standard deviation. Using global values.")
            lang_std_devs = None
        
        # Save the metrics in the results dictionary for each dataset
        metrics_results[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_overlap": avg_overlap,
            "samples_processed": samples_with_label_1,
            "individual_precision": individual_precision,
            "individual_recall": individual_recall,
            "individual_overlaps": overlap_scores,
            "std_devs": lang_std_devs
        }

    # Calculate global standard deviations
    global_std_devs = {
        "precision": np.std(all_precision_values) if all_precision_values else 0,
        "recall": np.std(all_recall_values) if all_recall_values else 0,
        "f1": np.std(all_metrics["f1"]),
        "avg_overlap": np.std(all_metrics["avg_overlap"]),
        "individual_overlap": np.std(all_overlap_scores) if all_overlap_scores else 0
    }
    
    # Calculate means for overall metrics
    means = {
        "precision": np.mean(all_precision_values) if all_precision_values else 0,
        "recall": np.mean(all_recall_values) if all_recall_values else 0,
        "f1": np.mean(all_metrics["f1"]),
        "avg_overlap": np.mean(all_metrics["avg_overlap"]),
        "individual_overlap": np.mean(all_overlap_scores) if all_overlap_scores else 0
    }

    save_metrics_to_tsv(
        'task2_metrics_with_span_matching.tsv',
        metrics_results,
        ['language_combination', 'precision', 'recall', 'f1', 'avg_overlap', 'samples_processed'],
        global_std_devs,
        means
    )
    
    print("Task 2 metrics have been saved.")
    return metrics_results, global_std_devs, means

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    template_file = "templates.json"
    demonstrations_file = "demonstrations.json"
    
    # Files to process
    input_files = ['en_L_filled_test.json', 'it_L_filled_test.json', 'pt_L_filled_test.json']
    
    try:
        # Verify files exist
        for file in [template_file, demonstrations_file] + input_files:
            if not os.path.exists(file):
                print(f"Warning: File {file} not found.")
        
        # Load templates and demonstrations
        templates = load_templates(template_file)
        demonstrations = load_demonstrations(demonstrations_file)
        
        if not templates:
            raise ValueError(f"No templates found in {template_file}")
        
        # Verify CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU instead.")
        
        # Load model and tokenizer
        print(f"Loading model and tokenizer: {model_name}")
        model, tokenizer, device = load_model_and_tokenizer(model_name)
        
        # Process all input files
        print("Processing input files...")
        all_data = process_all_files(input_files)
        
        # Generate text for all samples
        print("Generating text for all samples...")
        generated_data = generate_for_all_samples(
            all_data,
            templates,
            demonstrations,
            tokenizer,
            model,
            device
        )

        # After generating text for all samples:
        print("Computing metrics for Task 1 with standard deviation...")
        task1_results, task1_std_devs, task1_means = task1_metrics(generated_data)
        
        print("Computing metrics for Task 2 with standard deviation...")
        task2_results, task2_std_devs, task2_means = task2_metrics(generated_data)
        
        # Print overall metrics summary with standard deviation
        print("\nOVERALL METRICS SUMMARY WITH STANDARD DEVIATION")
        print("=" * 80)
        print(f"Task 1 - Accuracy: {task1_means['accuracy']:.4f} ± {task1_std_devs['accuracy']:.4f}")
        print(f"Task 1 - Precision: {task1_means['precision']:.4f} ± {task1_std_devs['precision']:.4f}")
        print(f"Task 1 - Recall: {task1_means['recall']:.4f} ± {task1_std_devs['recall']:.4f}")
        print(f"Task 1 - F1: {task1_means['f1']:.4f} ± {task1_std_devs['f1']:.4f}")
        print("-" * 80)
        print(f"Task 2 - Precision: {task2_means['precision']:.4f} ± {task2_std_devs['precision']:.4f}")
        print(f"Task 2 - Recall: {task2_means['recall']:.4f} ± {task2_std_devs['recall']:.4f}")
        print(f"Task 2 - F1: {task2_means['f1']:.4f} ± {task2_std_devs['f1']:.4f}")
        print(f"Task 2 - Average overlap: {task2_means['avg_overlap']:.4f} ± {task2_std_devs['avg_overlap']:.4f}")
        print(f"Task 2 - Individual sample overlap: {task2_means['individual_overlap']:.4f} ± {task2_std_devs['individual_overlap']:.4f}")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
