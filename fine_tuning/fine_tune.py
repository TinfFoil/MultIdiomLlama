import os
import json
from datetime import datetime
from collections import defaultdict
import logging
import torch
import numpy as np
import csv
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    AutoTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)
# Explicitly import all needed sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from prompter import Prompter

import logging

# Set up logging
logging.basicConfig(
    filename='training_metrics.log',  # Path to the log file
    level=logging.INFO,  # Log everything at INFO level and higher
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration class to centralize all parameters
class TrainingConfig:
    """
    Configuration class that centralizes all parameters for the training process.
    This includes model parameters, training hyperparameters, and dataset configurations.
    """
    def __init__(self):
        # Prompt template configuration
        self.prompt_template_name = "alpaca"  # Template for formatting prompts
        self.cutoff_len = 128  # Maximum sequence length for training
        
        # Training hyperparameters
        self.batch_size = 32  # Batch size for training and evaluation
        self.num_train_epochs = 2  # Number of training epochs
        self.output_dir = "/home/dciminari/thesis/results_08"  # Directory to save results
        self.device_map = "cuda:0"  # GPU device to use
        self.resume_from_checkpoint = None  # Path to checkpoint for resuming training
        self.mixed_precision = "fp16"  # Mixed precision training
        self.gradient_checkpointing = True  # Enable gradient checkpointing for memory efficiency
        
        # Dataset configuration
        self.val_set_size = 75000  # Size of validation dataset
        self.train_subset_fraction = 0.032  # Fraction of training data to use (3.2%)
        self.val_subset_fraction = 0.04  # Fraction of validation data to use (4%)
        self.test_subset_fraction = 0.04  # Fraction of test data to use (4%)
        
        # Model configuration
        self.base_model = "meta-llama/Llama-3.2-1B"  # Base model to fine-tune
        self.add_eos_token = True  # Whether to add EOS token to inputs
        
        # LoRA specific configurations for parameter-efficient fine-tuning
        self.lora_r = 8  # LoRA rank
        self.lora_alpha = 16  # LoRA alpha parameter
        self.lora_dropout = 0.05  # Dropout probability for LoRA layers
        self.lora_target_modules = ["q_proj", "k_proj"]  # Modules to apply LoRA to

        # Quantization config for memory efficiency
        self.bits = 4  # Quantization bit width
        self.double_quant = True  # Whether to use double quantization
        self.quant_type = "nf4"  # Quantization type (normal float 4-bit)


def initialize_tokenizer(model_name):
    """
    Initialize and configure the tokenizer with proper padding tokens.
    
    Args:
        model_name (str): The name or path of the pre-trained model
        
    Returns:
        tokenizer: Configured tokenizer ready for training
    """
    # Load the tokenizer from the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add new tokens to the vocabulary if needed
    new_tokens = ['!', '锦']
    tokenizer.add_tokens(new_tokens)

    # Log the updated vocabulary size
    print(f"Updated vocabulary size after adding new tokens: {len(tokenizer)}")

    # Set the pad_token if it is not already set
    if tokenizer.pad_token is None:
        # Define a new pad token (e.g., '[PAD]')
        new_pad_token = '[PAD]'
        # Make sure it's unique and not conflicting with eos or bos tokens
        tokenizer.add_tokens([new_pad_token])
        tokenizer.pad_token = new_pad_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(new_pad_token)
        print(f"Pad token set to: {tokenizer.pad_token} with id: {tokenizer.pad_token_id}")

    # Ensure the padding token is properly set
    if tokenizer.pad_token_id >= len(tokenizer):
        print(f"Warning: pad_token_id {tokenizer.pad_token_id} is out of vocabulary range!")
        tokenizer.pad_token_id = len(tokenizer) - 1  # Reset to the last token if necessary
        print(f"Reset pad_token_id to {tokenizer.pad_token_id}")

    # Set padding side to left for better performance with LLMs
    tokenizer.padding_side = "left"

    # Verify and log the tokenizer configuration
    print("Tokenizer configuration:")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"Pad token id: {tokenizer.pad_token_id}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"BOS token id: {tokenizer.bos_token_id}")
    print(f"BOS token: {tokenizer.bos_token}")
    print(f"EOS token id: {tokenizer.eos_token_id}")
    print(f"UNK token: {tokenizer.unk_token}")
    print(f"UNK token id: {tokenizer.unk_token_id}")
    print(f"Padding side: {tokenizer.padding_side}")

    return tokenizer

def initialize_model(config: TrainingConfig, tokenizer) -> LlamaForCausalLM:
    """
    Initialize the model with quantization and LoRA configurations.
    
    Args:
        config (TrainingConfig): Configuration object with model parameters
        tokenizer: Tokenizer to ensure model's token embeddings match vocabulary
        
    Returns:
        LlamaForCausalLM: Initialized and configured model ready for training
    """
    # Set up quantization config for 4-bit training
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config.double_quant,
        bnb_4bit_quant_type=config.quant_type
    )

    # Load base model with quantization for memory efficiency
    model = LlamaForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        device_map=config.device_map,
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_auth_token=os.getenv("HF_TOKEN")  # Use Hugging Face token from environment
    )

    # Resize token embeddings to match tokenizer vocabulary
    # This is necessary if new tokens were added to the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing for memory efficiency during training
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Prepare model for k-bit training with PEFT
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=config.lora_r,                        # Rank of LoRA update matrices
        lora_alpha=config.lora_alpha,           # LoRA scaling factor
        target_modules=config.lora_target_modules,  # Which modules to apply LoRA to
        lora_dropout=config.lora_dropout,       # Dropout probability
        bias="none",                            # Whether to train bias parameters
        task_type="CAUSAL_LM"                   # Task type (causal language modeling)
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Load checkpoint if exists to resume training
    if isinstance(config.resume_from_checkpoint, str):
        checkpoint_path = os.path.join(config.resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(config.resume_from_checkpoint, "adapter_model.bin")

        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            adapters_weights = torch.load(checkpoint_path)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"No checkpoint found at {config.resume_from_checkpoint}")

    # Print trainable parameters info for logging
    model.print_trainable_parameters()

    return model


# Create a simplified version of language detector similar to Prompter class
def detect_instruction_language(text):
    """
    Detect language from the instruction part of the text.
    
    This function analyzes the text to determine whether the instruction
    is in English (en), Italian (it), or Portuguese (pt) based on
    language-specific patterns and markers.
    
    Args:
        text (str): The full text including instruction and response
        
    Returns:
        str: Language code - 'en', 'it', or 'pt'
    """
    text_lower = text.lower()

    # Extract instruction part if possible
    instruction_text = ""
    if "### instruction:" in text_lower:
        parts = text_lower.split("### instruction:")
        if len(parts) > 1:
            # Look for end markers that separate instruction from other parts
            end_markers = ["### input:", "### response:", "### risposta:", "### resposta:"]
            instruction_part = parts[1]
            for marker in end_markers:
                if marker in instruction_part:
                    instruction_text = instruction_part.split(marker)[0]
                    break
            if not instruction_text:
                instruction_text = instruction_part
    else:
        # If can't extract instruction specifically, use the whole text
        instruction_text = text_lower

    # Look for language-specific instruction patterns
    en_patterns = ["identify", "find the idiom", "locate the idiom", "extract", "english"]
    it_patterns = ["individua", "trova", "identifica", "estrai", "italiano", "idiomatiche"]
    pt_patterns = ["identificar", "encontrar", "localizar", "extrair", "português", "idiomáticas"]

    # Count matches for each language
    en_count = sum(1 for pattern in en_patterns if pattern in instruction_text)
    it_count = sum(1 for pattern in it_patterns if pattern in instruction_text)
    pt_count = sum(1 for pattern in pt_patterns if pattern in instruction_text)

    # Return the language with most pattern matches
    counts = {"en": en_count, "it": it_count, "pt": pt_count}
    max_lang = max(counts, key=counts.get)

    # Default to English if no clear indicators
    if counts[max_lang] == 0:
        # Check response markers as fallback
        if "### response:" in text_lower:
            return "en"
        elif "### risposta:" in text_lower:
            return "it"
        elif "### resposta:" in text_lower:
            return "pt"
        return "en"  # Default to English if still no indicators

    return max_lang

class IdiomEvaluator:
    """
    Enhanced evaluator for idiom detection and extraction tasks with robust statistics.
    
    This class provides methods to evaluate model performance on two tasks:
    1. Task 1: Binary classification - detecting whether an idiom is present
    2. Task 2: Span identification - extracting the exact idiom text
    """

    def __init__(self):
        """Initialize evaluator with logging configuration"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def assign_label(self, text):
        """
        Determine if text contains an idiom (1) or not (0) based on specific markers in different languages.
        
        This function detects whether a response indicates the presence of an idiom or explicitly
        states that no idiom is present (using markers like "None", "Nessuna", etc.)
        
        Args:
            text (str): The response text to analyze
            
        Returns:
            int: 1 if an idiom is present, 0 if no idiom is present
        """
        if not text:
            return 0

        text_lower = text.lower()

        # Check for markers in different languages that indicate no idiom
        no_idiom_markers = [
            "none",                 # English
            "nessuna", "non",       # Italian
            "not", "not contain", "no idiom",  # English variations
            "não", "nenhuma", "ausente"        # Portuguese
        ]

        # Extract the response part
        response_part = ""
        for marker in ["### Response:", "### Risposta:", "### Resposta:"]:
            if marker in text:
                response_part = text.split(marker, 1)[1].strip().lower()
                break

        # If we have a response part, check it directly
        if response_part:
            for marker in no_idiom_markers:
                if marker in response_part:
                    return 0
            # If the response is empty or very short (just whitespace)
            if not response_part or len(response_part.strip()) == 0:
                return 0
            return 1

        # If no response part could be extracted, check the full text
        for marker in no_idiom_markers:
            if marker in text_lower:
                return 0

        # If no negative markers found, assume an idiom is present
        return 1

    def longest_common_subsequence(self, str1, str2):
        """
        Find the longest common subsequence between two strings.
        
        This allows for non-consecutive characters that maintain the same order,
        which is useful for partial matching of idioms where some characters
        might be different.
        
        Args:
            str1 (str): First string to compare
            str2 (str): Second string to compare
            
        Returns:
            tuple: (length of LCS, the subsequence string)
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

    def compute_overlap_with_metrics(self, gold_span, predicted_span):
        """
        Compute overlap metrics according to the partial text span matching formulas.
        
        The metrics calculated are:
        - Precision = |s∩t|/|t| (intersection / length of gold span)
        - Recall = |s∩t|/|s| (intersection / length of predicted span)
        - Overlap score = |s∩t|/min(|s|,|t|) (intersection / shorter span)
        
        Args:
            gold_span (str): The gold standard idiom span
            predicted_span (str): The predicted idiom span
            
        Returns:
            tuple: (precision, recall, overlap_score, common subsequence)
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
            return 0.0, 0.0, 0.0, ""  # One empty string, one non-empty string have 0% match

        # Compute the length of the longest common subsequence
        lcs_length, common_subsequence = self.longest_common_subsequence(gold_clean, predicted_clean)

        # Calculate metrics according to the formulas
        precision = lcs_length / len(gold_clean)      # |s∩t|/|t| where t is gold span
        recall = lcs_length / len(predicted_clean)    # |s∩t|/|s| where s is predicted span

        # Normalize by the length of the shorter of the two strings for the general overlap score
        overlap_score = lcs_length / min(len(gold_clean), len(predicted_clean))

        return precision, recall, overlap_score, common_subsequence

    def process_batch(self, decoded_labels, decoded_preds):
        """
        Process a batch of predictions and labels to extract input texts and idioms.
        
        This function handles the new format with consistent extraction of input and output fields.
        
        Args:
            decoded_labels (list): List of decoded gold standard texts
            decoded_preds (list): List of decoded prediction texts
            
        Returns:
            dict: Dictionary containing lists of input_texts, labels, predictions, and has_idiom flags
        """
        results = {
            "input_texts": [],
            "labels": [],
            "predictions": [],
            "has_idiom": []
        }

        for label, pred in zip(decoded_labels, decoded_preds):
            # Extract relevant parts from the text
            input_text = self._extract_input_text(label)
            gold_idiom = self._extract_idiom(label)
            pred_idiom = self._extract_idiom(pred)

            # Determine if the sample has an idiom
            has_idiom = 0 if gold_idiom.lower() in ["", "none", "nessuna", "nenhuma"] else 1

            # Store in results
            results["input_texts"].append(input_text)
            results["labels"].append(gold_idiom)
            results["predictions"].append(pred_idiom)
            results["has_idiom"].append(has_idiom)

        return results

    def _extract_input_text(self, text):
        """
        Extract the input text from the formatted prompt.
        
        Handles different language formats (English, Italian, Portuguese).
        
        Args:
            text (str): The full text containing inputs and responses
            
        Returns:
            str: The extracted input text
        """
        # Handle different languages in the prompts

        # English format
        if "### Input:" in text and "### Response:" in text:
            start = text.find("### Input:") + len("### Input:")
            end = text.find("### Response:")
            return text[start:end].strip()

        # Italian format
        if "### Input:" in text and "### Risposta:" in text:
            start = text.find("### Input:") + len("### Input:")
            end = text.find("### Risposta:")
            return text[start:end].strip()

        # Portuguese format
        if "### Input:" in text and "### Resposta:" in text:
            start = text.find("### Input:") + len("### Input:")
            end = text.find("### Resposta:")
            return text[start:end].strip()

        # If no specific format is found, try a generic approach
        for response_marker in ["### Response:", "### Risposta:", "### Resposta:"]:
            if response_marker in text:
                parts = text.split(response_marker)
                for input_marker in ["### Input:", "### Instruction:", "### Istruzione:", "### Instrução:"]:
                    if input_marker in parts[0]:
                        start = parts[0].find(input_marker) + len(input_marker)
                        return parts[0][start:].strip()

        # Fallback
        return ""

    def _extract_idiom(self, text):
        """
        Extract the idiom from the response.
        
        Handles different language formats and "None" responses.
        
        Args:
            text (str): The full text containing the response
            
        Returns:
            str: The extracted idiom, or empty string if none found
        """
        # Handle different languages in the prompts
        response_markers = ["### Response:", "### Risposta:", "### Resposta:"]

        for marker in response_markers:
            if marker in text:
                response_part = text.split(marker, 1)[1].strip()

                # Handle "None" and equivalent responses in different languages
                if response_part.lower() in ["none", "nessuna", "nenhuma"]:
                    return ""

                # Return the idiom
                return response_part

        # If we get here, no marker was found - try a fallback approach
        # Check if there are any lines after an empty line (common format)
        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            if not line.strip() and i+1 < len(lines):
                return lines[i+1].strip()

        # Last resort - return the whole text or just the last line
        return text.strip()

    def compute_metrics_task1(self, decoded_preds, decoded_labels, n_bootstrap=1000):
            """
            Compute Task 1 metrics with bootstrap resampling for binary idiom detection.
            
            Task 1 is the binary classification task of detecting whether an idiom is present.
            This method calculates accuracy, precision, recall, and F1 score, along with
            standard deviations using bootstrap resampling for reliability estimation.
            
            Args:
                decoded_preds (list): List of decoded model predictions
                decoded_labels (list): List of decoded gold standard labels
                n_bootstrap (int): Number of bootstrap iterations for standard deviation
                
            Returns:
                dict: Dictionary of metrics with standard deviations
            """
            self.logger.info(f"Computing Task 1 metrics for {len(decoded_preds)} samples")
            self.logger.info(f"Running bootstrap with {n_bootstrap} iterations")

            # Extract labels with error handling for each sample
            predicted_labels = []
            gold_labels = []

            for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
                try:
                    pred_label = self.assign_label(pred)
                    gold_label = self.assign_label(label)
                    predicted_labels.append(pred_label)
                    gold_labels.append(gold_label)
                except Exception as e:
                    self.logger.warning(f"Error assigning label for sample {i}: {str(e)}")
                    continue

            # Skip if no valid data
            if not predicted_labels or not gold_labels:
                self.logger.warning("No valid labels extracted for Task 1 metrics")
                return {
                    "accuracy_task1": 0,
                    "precision_task1": 0,
                    "recall_task1": 0,
                    "macro_f1_task1": 0,
                    "accuracy_std": 0,
                    "precision_std": 0,
                    "recall_std": 0,
                    "f1_std": 0
                }

            try:
                # Calculate base metrics
                acc = accuracy_score(gold_labels, predicted_labels)
                precision = precision_score(gold_labels, predicted_labels, zero_division=0)
                recall = recall_score(gold_labels, predicted_labels, zero_division=0)
                f1 = f1_score(gold_labels, predicted_labels, zero_division=0)

                # Log the base metrics every 50 steps
                for i in range(0, len(decoded_preds), 50):
                    if i % 50 == 0:
                        self.logger.info(f"Step {i}/{len(decoded_preds)} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

                # Create paired data for bootstrapping
                paired_data = list(zip(gold_labels, predicted_labels))

                # Bootstrap to calculate standard deviation
                bootstrap_accs = []
                bootstrap_precisions = []
                bootstrap_recalls = []
                bootstrap_f1s = []

                # Adjust bootstrap iterations for performance based on dataset size
                actual_n_bootstrap = max(100, min(n_bootstrap, 10000 // max(1, len(paired_data))))
                self.logger.info(f"Running {actual_n_bootstrap} bootstrap iterations for Task 1")

                for i in range(actual_n_bootstrap):
                    try:
                        # Sample with replacement from the paired data
                        bootstrap_indices = np.random.choice(len(paired_data), len(paired_data), replace=True)
                        bootstrap_sample = [paired_data[idx] for idx in bootstrap_indices]

                        # Calculate metrics for bootstrap sample
                        bootstrap_gold, bootstrap_pred = zip(*bootstrap_sample)
                        b_acc = accuracy_score(bootstrap_gold, bootstrap_pred)
                        b_precision = precision_score(bootstrap_gold, bootstrap_pred, zero_division=0)
                        b_recall = recall_score(bootstrap_gold, bootstrap_pred, zero_division=0)
                        b_f1 = f1_score(bootstrap_gold, bootstrap_pred, zero_division=0)

                        bootstrap_accs.append(b_acc)
                        bootstrap_precisions.append(b_precision)
                        bootstrap_recalls.append(b_recall)
                        bootstrap_f1s.append(b_f1)

                    except Exception as e:
                        self.logger.warning(f"Error in bootstrap iteration {i}: {str(e)}")
                        continue

                # Calculate standard deviations of the bootstrap distributions
                acc_std = np.std(bootstrap_accs) if bootstrap_accs else 0
                precision_std = np.std(bootstrap_precisions) if bootstrap_precisions else 0
                recall_std = np.std(bootstrap_recalls) if bootstrap_recalls else 0
                f1_std = np.std(bootstrap_f1s) if bootstrap_f1s else 0

                # Log results including standard deviations
                self.logger.info(f"Completed {len(bootstrap_accs)} bootstrap iterations")
                self.logger.info(f"Standard deviations - Acc: {acc_std:.4f}, Prec: {precision_std:.4f}, Recall: {recall_std:.4f}, F1: {f1_std:.4f}")
                self.logger.info(f"Task 1 metrics - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {acc:.4f}")

                # Return comprehensive metrics
                return {
                    "accuracy_task1": acc,
                    "precision_task1": precision,
                    "recall_task1": recall,
                    "macro_f1_task1": f1,
                    "accuracy_std": acc_std,
                    "precision_std": precision_std,
                    "recall_std": recall_std,
                    "f1_std": f1_std
                }

            except Exception as e:
                self.logger.error(f"Error computing Task 1 metrics: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {
                    "accuracy_task1": 0,
                    "precision_task1": 0,
                    "recall_task1": 0,
                    "macro_f1_task1": 0,
                    "accuracy_std": 0,
                    "precision_std": 0,
                    "recall_std": 0,
                    "f1_std": 0
                }


    def compute_metrics_task2(self, processed_batch, n_bootstrap=1000):
        """
        Compute Task 2 metrics for idiom span identification with bootstrap resampling.
        
        Task 2 evaluates how well the model extracts the exact idiom span.
        Only samples with idioms present are included in this evaluation.
        
        Args:
            processed_batch (dict): Batch of processed data with input_texts, labels, predictions, has_idiom
            n_bootstrap (int): Number of bootstrap iterations for standard deviation
            
        Returns:
            dict: Dictionary of metrics with standard deviations
        """
        self.logger.info(f"Running bootstrap with {n_bootstrap} iterations for Task 2")

        # Get data for samples with idioms only
        idiom_indices = [i for i, has_idiom in enumerate(processed_batch["has_idiom"]) if has_idiom == 1]

        if not idiom_indices:
            self.logger.warning("No samples with idioms found for Task 2 metrics")
            return {
                "macro_precision": 0,
                "macro_recall": 0,
                "macro_f1": 0,
                "precision_std": 0,
                "recall_std": 0,
                "f1_std": 0
            }

        # Extract relevant data for samples that contain idioms
        input_texts = [processed_batch["input_texts"][i] for i in idiom_indices]
        gold_idioms = [processed_batch["labels"][i] for i in idiom_indices]
        pred_idioms = [processed_batch["predictions"][i] for i in idiom_indices]

        self.logger.info(f"Computing Task 2 metrics for {len(idiom_indices)} samples with idioms")

        # Calculate individual metrics for each sample
        individual_precision = []
        individual_recall = []
        individual_overlap = []
        individual_data = []

        for i, (input_text, gold_idiom, pred_idiom) in enumerate(zip(input_texts, gold_idioms, pred_idioms)):
            precision, recall, overlap_score, _ = self.compute_overlap_with_metrics(gold_idiom, pred_idiom)

            individual_precision.append(precision)
            individual_recall.append(recall)
            individual_overlap.append(overlap_score)

            individual_data.append({
                'precision': precision,
                'recall': recall,
                'overlap_score': overlap_score
            })

        if not individual_precision:
            self.logger.warning("No valid precision values calculated for Task 2")
            return {
                "macro_precision": 0,
                "macro_recall": 0,
                "macro_f1": 0,
                "precision_std": 0,
                "recall_std": 0,
                "f1_std": 0
            }

        try:
            # Calculate aggregate metrics (macro averages)
            precision = sum(individual_precision) / len(individual_precision)
            recall = sum(individual_recall) / len(individual_recall)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Log the metrics every 50 steps
            for i in range(0, len(input_texts), 50):
                if i % 50 == 0:
                    self.logger.info(f"Step {i}/{len(input_texts)} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # Bootstrap to calculate standard deviations
            bootstrap_precisions = []
            bootstrap_recalls = []
            bootstrap_f1s = []

            # Adjust bootstrap iterations based on dataset size
            actual_n_bootstrap = max(100, min(n_bootstrap, 10000 // max(1, len(individual_data))))
            self.logger.info(f"Running {actual_n_bootstrap} bootstrap iterations for Task 2")

            for i in range(actual_n_bootstrap):
                try:
                    # Sample with replacement
                    bootstrap_indices = np.random.choice(len(individual_data), len(individual_data), replace=True)
                    bootstrap_sample = [individual_data[idx] for idx in bootstrap_indices]

                    # Calculate metrics for this bootstrap sample
                    b_precision = sum(item['precision'] for item in bootstrap_sample) / len(bootstrap_sample)
                    b_recall = sum(item['recall'] for item in bootstrap_sample) / len(bootstrap_sample)
                    b_f1 = 2 * (b_precision * b_recall) / (b_precision + b_recall) if (b_precision + b_recall) > 0 else 0

                    bootstrap_precisions.append(b_precision)
                    bootstrap_recalls.append(b_recall)
                    bootstrap_f1s.append(b_f1)

                except Exception as e:
                    self.logger.warning(f"Error in bootstrap iteration {i}: {str(e)}")
                    continue

            # Calculate standard deviations
            precision_std = np.std(bootstrap_precisions) if bootstrap_precisions else 0
            recall_std = np.std(bootstrap_recalls) if bootstrap_recalls else 0
            f1_std = np.std(bootstrap_f1s) if bootstrap_f1s else 0

            # Log results including standard deviations
            self.logger.info(f"Completed {len(bootstrap_precisions)} bootstrap iterations")
            self.logger.info(f"Standard deviations - Prec: {precision_std:.4f}, Recall: {recall_std:.4f}, F1: {f1_std:.4f}")
            self.logger.info(f"Task 2 metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            return {
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
                "precision_std": precision_std,
                "recall_std": recall_std,
                "f1_std": f1_std,
                "individual_metrics": individual_data
            }

        except Exception as e:
            self.logger.error(f"Error computing Task 2 metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "macro_precision": 0,
                "macro_recall": 0,
                "macro_f1": 0,
                "precision_std": 0,
                "recall_std": 0,
                "f1_std": 0
            }
        
    def save_metrics_to_file(self, metrics, file_prefix="metrics"):
        """
        Save metrics to a clean, tabular TSV file format with cross-language analysis.
        
        Creates a well-structured TSV file with columns for language, task, metric, value,
        and standard deviation, facilitating easy analysis in spreadsheet applications.
        
        Args:
            metrics (dict): Dictionary of metrics with nested structure
            file_prefix (str): Prefix for the output file name
        """
        try:
            # First, save the standard metrics file
            tsv_path = f'{file_prefix}.tsv'
            with open(tsv_path, 'w', newline='', encoding='utf-8') as tsv_file:
                # Define headers for the tabular format
                fieldnames = [
                    'language',
                    'task',
                    'metric',
                    'value',
                    'std_dev'
                ]
                writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()

                # Write metrics for each language in long format (one metric per row)
                for split in ["test", "validation"]:
                    if split in metrics:
                        if "by_instruction" in metrics[split]:
                            # New format with nested results
                            results_by_lang = metrics[split]["by_instruction"]
                        else:
                            # Old format with direct language keys
                            results_by_lang = metrics[split]

                        for lang in results_by_lang:
                            if lang not in ["overall", "std_dev"]:
                                # Task 1 metrics
                                if "task1" in results_by_lang[lang]:
                                    task1 = results_by_lang[lang]["task1"]
                                    for metric_name, value in task1.items():
                                        # Skip std metrics as they'll be in their own column
                                        if not metric_name.endswith("_std"):
                                            std_name = f"{metric_name}_std"
                                            std_value = task1.get(std_name, 0)

                                            writer.writerow({
                                                'language': lang,
                                                'task': 'task1',
                                                'metric': metric_name,
                                                'value': f"{value:.4f}",
                                                'std_dev': f"{std_value:.4f}"
                                            })

                                # Task 2 metrics
                                if "task2" in results_by_lang[lang]:
                                    task2 = results_by_lang[lang]["task2"]
                                    for metric_name, value in task2.items():
                                        # Skip individual metrics and std metrics
                                        if metric_name != "individual_metrics" and not metric_name.endswith("_std"):
                                            std_name = f"{metric_name}_std"
                                            std_value = task2.get(std_name, 0)

                                            writer.writerow({
                                                'language': lang,
                                                'task': 'task2',
                                                'metric': metric_name,
                                                'value': f"{value:.4f}",
                                                'std_dev': f"{std_value:.4f}"
                                            })

                        # Write overall metrics
                        if "overall" in results_by_lang:
                            for task_name in ["task1", "task2"]:
                                if task_name in results_by_lang["overall"]:
                                    task_metrics = results_by_lang["overall"][task_name]
                                    for metric_name, value in task_metrics.items():
                                        if not metric_name.endswith("_std"):
                                            writer.writerow({
                                                'language': 'OVERALL',
                                                'task': task_name,
                                                'metric': metric_name,
                                                'value': f"{value:.4f}",
                                                'std_dev': "0.0000"
                                            })

            self.logger.info(f"Metrics have been saved to {tsv_path}")

            # Now save cross-language analysis if available
            if "test" in metrics and "by_instruction_input" in metrics["test"]:
                cross_lang_metrics = metrics["test"]["by_instruction_input"]

                # Get unique instruction and input languages
                instruction_langs = set()
                input_langs = set()

                for lang_pair in cross_lang_metrics.keys():
                    if "_" in lang_pair:
                        instruct_lang, input_lang = lang_pair.split("_", 1)
                        instruction_langs.add(instruct_lang)
                        input_langs.add(input_lang)

                # Create sorted lists
                instruction_langs = sorted(list(instruction_langs))
                input_langs = sorted(list(input_langs))

                # Save matrix for Task 1 F1 score
                self._save_metric_matrix(
                    cross_lang_metrics,
                    instruction_langs,
                    input_langs,
                    "task1",
                    "macro_f1_task1",
                    os.path.join(os.path.dirname(file_prefix), "task1_f1_matrix.tsv")
                )

                # Save matrix for Task 2 F1 score
                self._save_metric_matrix(
                    cross_lang_metrics,
                    instruction_langs,
                    input_langs,
                    "task2",
                    "macro_f1",
                    os.path.join(os.path.dirname(file_prefix), "task2_f1_matrix.tsv")
                )

                self.logger.info("Cross-language metric matrices have been saved")

        except Exception as e:
            self.logger.error(f"Error saving metrics to file: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _extract_response_only(self, text):
        """
        Extract only the portion after response markers.
        
        Args:
            text (str): The full text containing a response section
            
        Returns:
            str: The extracted response text
        """
        for marker in ["### Response:", "### Risposta:", "### Resposta:"]:
            if marker in text:
                return text.split(marker, 1)[1].strip()
        
        # If no marker found, return empty string
        return ""

    def _collect_examples(self, preds, labels, processed_batch, limit=10):
        """
        Collect a limited number of examples with metrics for reference.
        
        Args:
            preds (list): List of model predictions
            labels (list): List of gold standard labels
            processed_batch (dict): Pre-processed batch with extracted idioms
            limit (int): Maximum number of examples to collect
            
        Returns:
            list: Examples with metrics for both tasks
        """
        examples = []

        for i, (pred, label) in enumerate(zip(preds, labels)):
            # Get response-only parts
            gold_response = self._extract_response_only(label)
            pred_response = self._extract_response_only(pred)

            # Get Task 1 metrics
            pred_label = self.assign_label(pred)
            gold_label = self.assign_label(label)
            task1_correct = pred_label == gold_label

            # Get Task 2 metrics if there's an idiom
            task2_score = 0
            if gold_label == 1 and i < len(processed_batch["labels"]):  # If there's an idiom
                precision, recall, overlap, _ = self.compute_overlap_with_metrics(
                    processed_batch["labels"][i],
                    processed_batch["predictions"][i]
                )
                task2_score = overlap

            examples.append({
                "gold_text": label,
                "pred_text": pred,
                "gold_response": gold_response,
                "pred_response": pred_response,
                "input_text": self._extract_input_text(label),
                "task1_correct": task1_correct,
                "task2_score": task2_score
            })

            if len(examples) >= limit:
                break

        return examples

    def evaluate_by_language(self, decoded_preds, decoded_labels, test_data=None, val_data=None, output_dir=None):
        """
        Evaluate metrics separated by instruction language and instruction-input language pairs.
        
        This comprehensive evaluation function analyzes performance across different language
        combinations, providing insight into cross-lingual transfer capabilities.
        
        Args:
            decoded_preds (list): List of decoded model predictions
            decoded_labels (list): List of decoded gold standard labels
            test_data: Original test data JSON or Dataset object with language metadata
            val_data: Validation dataset for additional metrics
            output_dir: Directory to save output files
            
        Returns:
            dict: Nested dictionary of metrics by language combinations
        """
        # Get language groups by filtering predictions
        lang_groups = self._filter_predictions_by_language(decoded_preds, decoded_labels, test_data)
        
        results_by_lang = {}  # Results by instruction language only
        results_by_lang_pair = {}  # Results by instruction-input language pair
        wrong_samples = {"task1": {}, "task2": {}}  # Track wrong predictions for analysis

        # First, evaluate by instruction language only
        for lang, indices in lang_groups["by_instruction"].items():
            if not indices:
                continue

            # Extract language-specific data
            lang_preds = [decoded_preds[i] for i in indices]
            lang_labels = [decoded_labels[i] for i in indices]

            # Process batch
            processed_batch = self.process_batch(lang_labels, lang_preds)

            # Compute metrics
            task1_metrics = self.compute_metrics_task1(lang_preds, lang_labels)
            task2_metrics = self.compute_metrics_task2(processed_batch)

            # Store results
            results_by_lang[lang] = {
                "task1": task1_metrics,
                "task2": task2_metrics,
                "examples": self._collect_examples(lang_preds, lang_labels, processed_batch, limit=10),
                "sample_count": len(indices)  # Track sample count
            }

            # Collect wrong samples for analysis
            wrong_samples["task1"][lang] = self._collect_wrong_task1_samples(lang_preds, lang_labels)
            wrong_samples["task2"][lang] = self._collect_wrong_task2_samples(processed_batch)

        # Next, evaluate by instruction-input language pairs
        for lang_pair, indices in lang_groups["by_instruction_input"].items():
            if not indices:
                continue

            # Extract data for this language pair
            pair_preds = [decoded_preds[i] for i in indices]
            pair_labels = [decoded_labels[i] for i in indices]

            # Process batch
            processed_batch = self.process_batch(pair_labels, pair_preds)

            # Compute metrics
            task1_metrics = self.compute_metrics_task1(pair_preds, pair_labels)
            task2_metrics = self.compute_metrics_task2(processed_batch)

            # Store results with sample count
            results_by_lang_pair[lang_pair] = {
                "task1": task1_metrics,
                "task2": task2_metrics,
                "sample_count": len(indices)  # Track sample count for each language pair
            }

        # Calculate overall results across instruction languages (simple average)
        if results_by_lang:
            overall = {
                "task1": {
                    "accuracy_task1": np.mean([results_by_lang[lang]["task1"]["accuracy_task1"] for lang in results_by_lang]),
                    "precision_task1": np.mean([results_by_lang[lang]["task1"]["precision_task1"] for lang in results_by_lang]),
                    "recall_task1": np.mean([results_by_lang[lang]["task1"]["recall_task1"] for lang in results_by_lang]),
                    "macro_f1_task1": np.mean([results_by_lang[lang]["task1"]["macro_f1_task1"] for lang in results_by_lang])
                },
                "task2": {
                    "macro_precision": np.mean([results_by_lang[lang]["task2"]["macro_precision"] for lang in results_by_lang]),
                    "macro_recall": np.mean([results_by_lang[lang]["task2"]["macro_recall"] for lang in results_by_lang]),
                    "macro_f1": np.mean([results_by_lang[lang]["task2"]["macro_f1"] for lang in results_by_lang])
                },
                "sample_count": sum([results_by_lang[lang]["sample_count"] for lang in results_by_lang])
            }
            results_by_lang["overall"] = overall

        # Save detailed metrics to TSV files
        if output_dir:
            self._save_cross_language_metrics(results_by_lang_pair, output_dir)
            with open(f'{output_dir}/wrong_samples.json', 'w', encoding='utf-8') as f:
                json.dump(wrong_samples, f, indent=4, ensure_ascii=False)

        return {
            "by_instruction": results_by_lang,
            "by_instruction_input": results_by_lang_pair,
            "wrong_samples": wrong_samples
        }
    
    def _filter_predictions_by_language(self, decoded_preds, decoded_labels, test_data):
        """
        Group prediction indices by both instruction language and input language.
        
        This method specifically handles the format where input fields have language 
        indicators (input_en, input_pt, etc.) and creates two groupings:
        1. By instruction language only
        2. By instruction-input language pairs
        
        Args:
            decoded_preds (list): List of decoded predictions
            decoded_labels (list): List of decoded labels
            test_data: Original test data JSON or Dataset object
            
        Returns:
            dict: Dictionary with two groupings: "by_instruction" and "by_instruction_input"
        """
        lang_groups = defaultdict(list)
        instruction_input_groups = defaultdict(list)

        # Convert Dataset object to list if needed
        if hasattr(test_data, 'to_list'):
            self.logger.info("Converting Dataset object to list")
            test_data = test_data.to_list()
        elif hasattr(test_data, 'to_pandas'):
            self.logger.info("Converting Dataset object to list via pandas")
            test_data = test_data.to_pandas().to_dict('records')

        # Check test_data format
        if isinstance(test_data, list):
            # For each example in the test set, determine its languages
            for idx, example in enumerate(test_data):
                # First, determine the instruction language
                instruction = example.get("instruction", "")
                instruct_lang = example.get("instruct_lang", "")

                if not instruct_lang:
                    instruct_lang = self._detect_language_from_content(instruction)

                # Next, determine the input language from the input_XX field name
                input_lang = ""
                for key in example.keys():
                    if key.startswith("input_") and example[key]:
                        input_lang = key.split("_")[1]
                        break

                # If no input language found, use a default
                if not input_lang:
                    input_lang = "unknown"

                # Store by instruction language for traditional grouping
                lang_groups[instruct_lang].append(idx)

                # Store by instruction-input language pair for detailed analysis
                lang_pair = f"{instruct_lang}_{input_lang}"
                instruction_input_groups[lang_pair].append(idx)

                # Also store by metadata if available (from the dataset processing)
                if "metadata" in example and isinstance(example["metadata"], dict):
                    meta_instruct_lang = example["metadata"].get("instruct_lang")
                    meta_input_lang = example["metadata"].get("input_lang")

                    if meta_instruct_lang and meta_input_lang:
                        meta_lang_pair = f"{meta_instruct_lang}_{meta_input_lang}"
                        # Only add if different from what we already determined
                        if meta_lang_pair != lang_pair:
                            instruction_input_groups[meta_lang_pair].append(idx)
        else:
            # If test_data is not a list or conversion failed
            self.logger.warning("test_data is not a list, unable to perform language-based grouping")
            lang_groups["unknown"] = list(range(len(decoded_preds)))
            instruction_input_groups["unknown_unknown"] = list(range(len(decoded_preds)))

        # Log distribution of languages
        for lang, indices in lang_groups.items():
            self.logger.info(f"Detected {len(indices)} samples with instruction language '{lang}'")

        for lang_pair, indices in instruction_input_groups.items():
            self.logger.info(f"Detected {len(indices)} samples with instruction-input language pair '{lang_pair}'")

        # Return both groupings
        return {
            "by_instruction": lang_groups,
            "by_instruction_input": instruction_input_groups
        }

    def _detect_language_from_content(self, text):
        """
        Detect language from text content as a fallback when instruct_lang is not available.
        
        Analyzes the text for language-specific patterns to determine if it's
        English, Italian, or Portuguese.
        
        Args:
            text (str): The text to analyze for language markers
            
        Returns:
            str: Detected language code ('en', 'it', or 'pt')
        """
        text_lower = text.lower()

        # Look for language-specific instruction patterns
        en_patterns = ["identify", "find the idiom", "locate the idiom", "extract", "english", "analysis", "can you spot"]
        it_patterns = ["individua", "trova", "identifica", "estrai", "italiano", "idiomatiche", "riuscite"]
        pt_patterns = ["identificar", "encontrar", "localizar", "extrair", "português", "idiomáticas"]

        # Count matches for each language
        en_count = sum(1 for pattern in en_patterns if pattern in text_lower)
        it_count = sum(1 for pattern in it_patterns if pattern in text_lower)
        pt_count = sum(1 for pattern in pt_patterns if pattern in text_lower)

        # Return the language with most pattern matches
        counts = {"en": en_count, "it": it_count, "pt": pt_count}
        max_lang = max(counts, key=counts.get)

        # Default to English if no clear indicators
        if counts[max_lang] == 0:
            return "en"

        return max_lang

    def _save_cross_language_metrics(self, results_by_lang_pair, output_dir):
        """
        Save metrics in a dataframe-like TSV format with instruction languages as rows
        and input languages as columns.
        
        Creates separate files for different metrics to facilitate cross-lingual analysis.
        
        Args:
            results_by_lang_pair (dict): Results by language pair
            output_dir (str): Directory to save output files
        """
        metrics_to_save = [
            "accuracy_task1", "precision_task1", "recall_task1", "macro_f1_task1",  # Task 1 metrics
            "macro_precision", "macro_recall", "macro_f1"  # Task 2 metrics
        ]

        # Extract unique instruction and input languages
        instruction_langs = set()
        input_langs = set()

        for lang_pair in results_by_lang_pair.keys():
            if "_" in lang_pair:
                instruct_lang, input_lang = lang_pair.split("_", 1)
                instruction_langs.add(instruct_lang)
                input_langs.add(input_lang)

        # Create sorted lists
        instruction_langs = sorted(list(instruction_langs))
        input_langs = sorted(list(input_langs))

        # Create separate TSV files for each metric
        for metric_name in metrics_to_save:
            task = "task1" if "task1" in metric_name else "task2"
            metric_key = metric_name

            tsv_path = os.path.join(output_dir, f"cross_language_{metric_name}.tsv")

            with open(tsv_path, 'w', newline='', encoding='utf-8') as tsv_file:
                writer = csv.writer(tsv_file, delimiter='\t')

                # Write header row with input languages
                header = ["Instruction\\Input"] + input_langs
                writer.writerow(header)

                # Write a row for each instruction language
                for instruct_lang in instruction_langs:
                    row = [instruct_lang]

                    for input_lang in input_langs:
                        lang_pair = f"{instruct_lang}_{input_lang}"

                        if lang_pair in results_by_lang_pair and task in results_by_lang_pair[lang_pair]:
                            value = results_by_lang_pair[lang_pair][task].get(metric_key, 0)
                            row.append(f"{value:.4f}")
                        else:
                            row.append("N/A")

                    writer.writerow(row)

            self.logger.info(f"Saved cross-language {metric_name} matrix to {tsv_path}")

        # Create a summary TSV with F1 scores for both tasks
        summary_path = os.path.join(output_dir, "cross_language_summary.tsv")
        with open(summary_path, 'w', newline='', encoding='utf-8') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')

            # Write headers
            writer.writerow(["Metric", "Instruction Lang", "Input Lang", "Value"])

            # Write F1 scores for both tasks
            for task, metric in [("task1", "macro_f1_task1"), ("task2", "macro_f1")]:
                for instruct_lang in instruction_langs:
                    for input_lang in input_langs:
                        lang_pair = f"{instruct_lang}_{input_lang}"

                        if lang_pair in results_by_lang_pair and task in results_by_lang_pair[lang_pair]:
                            value = results_by_lang_pair[lang_pair][task].get(metric, 0)
                            writer.writerow([f"F1_{task}", instruct_lang, input_lang, f"{value:.4f}"])

        self.logger.info(f"Saved cross-language summary to {summary_path}")


    def _save_metric_matrix(self, metrics, instruction_langs, input_langs, task, metric_name, output_path):
        """
        Save a specific metric as a matrix with instruction languages as rows and input languages as columns.
        
        Args:
            metrics (dict): Dictionary of metrics by language pair
            instruction_langs (list): List of instruction languages
            input_langs (list): List of input languages
            task (str): Task name (task1 or task2)
            metric_name (str): Name of the metric to save
            output_path (str): Path to save the output file
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as tsv_file:
                writer = csv.writer(tsv_file, delimiter='\t')

                # Write header row with input languages
                header = ["Instruction\\Input"] + input_langs
                writer.writerow(header)

                # Write a row for each instruction language
                for instruct_lang in instruction_langs:
                    row = [instruct_lang]

                    for input_lang in input_langs:
                        lang_pair = f"{instruct_lang}_{input_lang}"

                        if lang_pair in metrics and task in metrics[lang_pair]:
                            value = metrics[lang_pair][task].get(metric_name, 0)
                            row.append(f"{value:.4f}")
                        else:
                            row.append("N/A")

                    writer.writerow(row)

        except Exception as e:
            self.logger.error(f"Error saving metric matrix to {output_path}: {e}")

class CustomTrainer(Trainer):
    """
    Custom Trainer class that extends Hugging Face's Trainer with enhanced functionality.
    
    This custom implementation provides:
    1. Better handling of tokenizer padding during training and evaluation
    2. Enhanced evaluation with language-specific metrics
    3. Improved prediction decoding and error handling
    4. Language-based result grouping
    5. Periodic metrics saving
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the custom trainer with additional functionality.
        
        Args:
            *args: Positional arguments for the parent Trainer class
            **kwargs: Keyword arguments, should include 'tokenizer'
        """
        # Extract and validate tokenizer
        self.tokenizer = kwargs.get('tokenizer', None)
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")

        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Save the original padding side for reference
        self.original_padding_side = self.tokenizer.padding_side

        super().__init__(*args, **kwargs)

        # Generation parameters for text generation during evaluation
        self.generation_kwargs = {
            "max_new_tokens": 64,      # Maximum number of new tokens to generate
            "num_beams": 3,            # Beam search for better quality outputs
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,         # Use sampling for more diverse outputs
            "temperature": 0.4,        # Lower temperature for more focused outputs
            "top_p": 0.9,              # Nucleus sampling parameter
            "early_stopping": True     # Stop when EOS token is generated
        }
        
        # Add metrics saving functionality
        self.metrics_save_interval = 1000  # Save metrics every 1000 steps

        # Setup logging handlers and metrics directories
        self._setup_logging()
        self._create_metrics_directories()
        self.evaluator = IdiomEvaluator()
        
    def training_step(self, model, inputs, num_items_in_batch, **kwargs):
        """
        Perform a training step with periodic metrics saving.
        
        Args:
            model: The model to train
            inputs: Input tensors
            num_items_in_batch: Number of items in the batch
            **kwargs: Additional arguments
            
        Returns:
            Output from the training step
        """
        # Perform the regular training step
        step_output = super().training_step(model, inputs)
        print("Training step called!")
        
        # Return the step output
        return step_output

    def _setup_logging(self):
        """
        Set up logging with both file and console handlers.
        
        Configures a dedicated log file for this training run with timestamp
        and also logs to console for real-time monitoring.
        """
        log_dir = os.path.join(self.args.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.handlers = []
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _create_metrics_directories(self):
        """
        Create directories for storing metrics and evaluation results.
        
        Sets up file paths for metrics logging and detailed metrics storage.
        """
        self.metrics_log_file = os.path.join(self.args.output_dir, 'metrics.json')
        self.detailed_metrics_dir = os.path.join(self.args.output_dir, 'detailed_metrics')
        os.makedirs(self.detailed_metrics_dir, exist_ok=True)

    def train(self, *args, **kwargs):
        """
        Ensure left padding for training and run the training process.
        
        Left padding is better for causal language modeling as it places the
        attention on the right side of the sequence (the part to predict).
        
        Args:
            *args: Positional arguments for parent train method
            **kwargs: Keyword arguments for parent train method
            
        Returns:
            Training result from parent train method
        """
        # Set padding to left for training
        self.tokenizer.padding_side = "left"
        self.logger.info(f"Setting tokenizer padding_side to 'left' for training")

        # Run training
        result = super().train(*args, **kwargs)
        return result

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Single source of truth for generating predictions with attention mask logging.
        
        This method handles the generation of text during evaluation, with
        careful attention to masking and proper logging for debugging.
        
        Args:
            model: The model to use for predictions
            inputs: Input tensors
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore in the model output
            
        Returns:
            tuple: (loss, predictions, labels)
        """
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Debug: Print attention mask for the first example
        if 'attention_mask' in inputs:
            print("Sample attention mask:", inputs['attention_mask'][0].cpu().numpy())
            # You can also print the sum to see how many tokens are attended to
            print("Number of attended tokens:", inputs['attention_mask'][0].sum().item())

        with torch.no_grad():
            # Get loss
            outputs = model(**inputs)
            loss = outputs.loss.mean().detach() if outputs.loss is not None else None

            # Generate predictions using model.generate() with configured parameters
            predictions = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **self.generation_kwargs
            )

            # Debug logging
            if self.logger.isEnabledFor(logging.DEBUG):
                sample_idx = 0  # First example in batch
                sample_input = self.tokenizer.decode(inputs["input_ids"][sample_idx])
                sample_pred = self.tokenizer.decode(predictions[sample_idx])
                self.logger.debug(f"Sample input: {sample_input[:100]}...")
                self.logger.debug(f"Sample raw prediction: {sample_pred[:100]}...")

        return (loss, predictions, inputs["labels"])

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", test_data=None, val_data=None, output_dir=None):
        """
        Evaluate the model on eval_dataset with enhanced logging for language combinations.
        
        This extended evaluation method performs language-specific evaluation,
        logging detailed metrics by language, and saving comprehensive results.
        
        Args:
            eval_dataset: Evaluation dataset
            ignore_keys: Keys to ignore in the model output
            metric_key_prefix: Prefix for metric names
            test_data: Original test data in JSON format for language filtering
            val_data: Validation data for additional metrics
            output_dir: Path to save evaluation results
            
        Returns:
            dict: Evaluation metrics
        """
        # Set padding to right for evaluation/testing
        self.tokenizer.padding_side = "right"
        self.logger.info(f"Setting tokenizer padding_side to 'right' for evaluation")

        # Use the dataset passed or the eval dataset from initialization
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # Verify the tokenizer is properly configured
        self._check_processor_config()

        # Run predictions
        output = self.predict(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        predictions, labels = output.predictions, output.label_ids

        # Process predictions and labels
        try:
            decoded_preds, decoded_labels = self._decode_predictions_labels(predictions, labels)

            # Check for language-specific evaluation
            if test_data:
                # Log evaluation step
                self.logger.info(f"=" * 80)
                self.logger.info(f"EVALUATION AT STEP {self.state.global_step}")
                self.logger.info(f"=" * 80)
                
                evaluation_results = self.evaluator.evaluate_by_language(
                    decoded_preds, decoded_labels, test_data, val_data, output_dir
                )
                results_by_lang = evaluation_results["by_instruction"]
                wrong_samples = evaluation_results["wrong_samples"]

                # Validate on validation set if provided
                validation_results = None
                if val_data:
                    # Run predictions on validation set
                    val_output = self.predict(val_data, ignore_keys=ignore_keys, metric_key_prefix="val")
                    val_preds, val_labels = val_output.predictions, val_output.label_ids
                    val_decoded_preds, val_decoded_labels = self._decode_predictions_labels(val_preds, val_labels)

                    # Process validation batch
                    val_batch = self.evaluator.process_batch(val_decoded_labels, val_decoded_preds)
                    validation_results = self.evaluator.compute_metrics_task2(val_batch)

                # Combine results
                final_results = {
                    "test": results_by_lang,
                    "validation": {"task2": validation_results} if validation_results else {}
                }

                # Save results if filepath provided
                if output_dir:
                    # Create timestamped files to avoid overwriting
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    metrics_dir = os.path.join(output_dir, 'metrics')
                    os.makedirs(metrics_dir, exist_ok=True)
                    
                    # Save evaluation results with step number and timestamp
                    metrics_file = os.path.join(metrics_dir, f'eval_results_step{self.state.global_step}_{timestamp}.json')
                    with open(metrics_file, 'w', encoding='utf-8') as f:
                        json.dump({"results": final_results, "wrong_samples": wrong_samples}, f, indent=4, ensure_ascii=False)

                    # Also save TSV files for easier analysis
                    tsv_dir = os.path.join(output_dir, f'tsvs_step{self.state.global_step}')
                    os.makedirs(tsv_dir, exist_ok=True)
                    self.evaluator.save_metrics_to_file(final_results, os.path.join(tsv_dir, "evaluation_results"))

                # Restore original padding side
                self.tokenizer.padding_side = self.original_padding_side

                # Return the F1 scores for both tasks in the overall results for use in best model selection
                if "overall" in results_by_lang:
                    return {
                        f"{metric_key_prefix}_task1_f1": results_by_lang["overall"]["task1"]["macro_f1_task1"],
                        f"{metric_key_prefix}_task2_f1": results_by_lang["overall"]["task2"]["macro_f1"]
                    }
                else:
                    # Compute simple averages across languages if no overall results available
                    task1_f1 = np.mean([results_by_lang[lang]["task1"]["macro_f1_task1"] for lang in results_by_lang if lang != "overall"])
                    task2_f1 = np.mean([results_by_lang[lang]["task2"]["macro_f1"] for lang in results_by_lang if lang != "overall"])
                    return {
                        f"{metric_key_prefix}_task1_f1": task1_f1,
                        f"{metric_key_prefix}_task2_f1": task2_f1
                    }

            else:
                # Standard evaluation without language filtering
                metrics = {}

                # Compute metrics for Task 1
                task1_metrics = self.evaluator.compute_metrics_task1(decoded_preds, decoded_labels)
                metrics.update({f"{metric_key_prefix}_task1_{k}": v for k, v in task1_metrics.items()})

                # Compute metrics for Task 2
                processed_batch = self.evaluator.process_batch(decoded_labels, decoded_preds)
                task2_metrics = self.evaluator.compute_metrics_task2(processed_batch)
                metrics.update({f"{metric_key_prefix}_task2_{k}": v for k, v in task2_metrics.items()})

                # Restore original padding side
                self.tokenizer.padding_side = self.original_padding_side

                return metrics
        except Exception as e:
            # Restore original padding side even when there's an error
            self.tokenizer.padding_side = self.original_padding_side
            self.logger.error(f"Error during evaluation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _decode_predictions_labels(self, predictions, labels):
        """
        Decode predictions and labels to text with proper token handling.
        
        This method carefully handles token ID clipping, problematic tokens,
        and special tokens to ensure reliable decoding of model outputs.
        
        Args:
            predictions: Predictions from model (token IDs)
            labels: Gold standard labels (token IDs)
            
        Returns:
            tuple: (decoded_preds, decoded_labels) as text
        """
        try:
            # Convert to numpy arrays if needed
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            if hasattr(labels, 'numpy'):
                labels = labels.numpy()

            # If predictions are logits, convert to token IDs
            if len(predictions.shape) > 2:
                predictions = np.argmax(predictions, axis=-1)

            # Process labels - replace padding tokens (-100)
            labels = np.where(labels == -100, self.tokenizer.pad_token_id, labels)

            # Use the actual tokenizer length, not vocabulary_size
            max_vocab_id = len(self.tokenizer) - 1
            self.logger.info(f"Max valid token ID: {max_vocab_id}")

            # Clip to valid token ID range
            predictions = np.clip(predictions, 0, max_vocab_id)
            labels = np.clip(labels, 0, max_vocab_id)

            # Add a safety check to filter out potentially problematic token IDs
            # This is a more aggressive approach to handle the issue
            problematic_ids = [128000, 128256, 128001]  # Add any other problematic token IDs here
            for pid in problematic_ids:
                predictions = np.where(predictions == pid, self.tokenizer.pad_token_id, predictions)
                labels = np.where(labels == pid, self.tokenizer.pad_token_id, labels)

            # Ensure int64 dtype for tokenizer compatibility
            predictions = predictions.astype(np.int64)
            labels = labels.astype(np.int64)

            # Batch decode with more explicit handling of special tokens
            decoded_preds = []
            decoded_labels = []

            # Decode each sequence individually for better control
            for pred_seq in predictions:
                # Skip any problematic token IDs
                valid_seq = [tid for tid in pred_seq if 0 <= tid < self.tokenizer.vocab_size and tid not in problematic_ids]
                if valid_seq:
                    decoded_preds.append(self.tokenizer.decode(valid_seq, skip_special_tokens=True))
                else:
                    decoded_preds.append("")

            for label_seq in labels:
                # Skip any problematic token IDs
                valid_seq = [tid for tid in label_seq if 0 <= tid < self.tokenizer.vocab_size and tid not in problematic_ids]
                if valid_seq:
                    decoded_labels.append(self.tokenizer.decode(valid_seq, skip_special_tokens=True))
                else:
                    decoded_labels.append("")

            # Clean up any remaining special tokens that might not be caught
            for i in range(len(decoded_preds)):
                # Remove any fragments of special tokens that remain
                for token in ['<s>', '</s>', '<pad>', '<|endoftext|>', '<|end_of_text|>']:
                    decoded_preds[i] = decoded_preds[i].replace(token, '')

            for i in range(len(decoded_labels)):
                for token in ['<s>', '</s>', '<pad>', '<|endoftext|>', '<|end_of_text|>']:
                    decoded_labels[i] = decoded_labels[i].replace(token, '')

            return decoded_preds, decoded_labels

        except Exception as e:
            self.logger.error(f"Error during decoding: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

            # Fallback decoding for troubleshooting
            decoded_preds = []
            decoded_labels = []

            for pred, label in zip(predictions, labels):
                try:
                    decoded_preds.append(self.tokenizer.decode(pred, skip_special_tokens=True))
                    decoded_labels.append(self.tokenizer.decode(label, skip_special_tokens=True))
                except Exception as e:
                    self.logger.error(f"Error decoding individual sample: {e}")
                    decoded_preds.append("")
                    decoded_labels.append("")

            return decoded_preds, decoded_labels

    def _check_processor_config(self):
        """
        Verify tokenizer configuration.
        
        Ensures tokenizer is properly configured with appropriate pad token,
        vocabulary size, and other essential parameters.
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer is not set.")

        self.logger.info(f"Tokenizer vocabulary size: {self.tokenizer.vocab_size}")
        self.logger.info(f"Pad token ID: {self.tokenizer.pad_token_id}")
        self.logger.info(f"EOS token ID: {self.tokenizer.eos_token_id}")
        self.logger.info(f"UNK token ID: {self.tokenizer.unk_token_id}")
        self.logger.info(f"Padding side: {self.tokenizer.padding_side}")

        # Ensure pad_token_id is set
        if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
            self.logger.warning("Tokenizer has no pad_token_id set. Setting to eos_token_id.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


def tokenize(prompt, tokenizer, config):
    """
    Tokenize the prompt with proper padding and truncation.
    
    Args:
        prompt (str): The text prompt to tokenize
        tokenizer: The tokenizer to use
        config (TrainingConfig): Configuration with parameters
        
    Returns:
        dict: Tokenized inputs
    """
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=config.cutoff_len,
        padding='max_length',
        add_special_tokens=True,
        return_tensors=None,
    )

    # Ensure proper token handling
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return result

def generate_and_tokenize_prompt(data_point, prompter, tokenizer, config, is_training=True):
    """
    Generate and tokenize prompt from data point using the updated format with input_lang fields.
    
    This function handles multilingual input fields (input_en, input_pt, input_it)
    and creates appropriately formatted prompts with language detection.
    
    Args:
        data_point (dict): The data point containing instruction and input_lang fields
        prompter (Prompter): The prompter to use (multilingual)
        tokenizer: The tokenizer to use
        config (TrainingConfig): Configuration object
        is_training (bool): Whether this is for training or evaluation
        
    Returns:
        dict: Tokenized prompt with metadata
    """
    # Get instruction
    instruction = data_point.get("instruction", "")

    # Detect which input field contains the language (input_en, input_pt, input_it)
    input_text = ""
    input_lang = ""

    for key in data_point.keys():
        if key.startswith("input_") and data_point[key]:
            input_text = data_point[key]
            input_lang = key.split("_")[1]  # Extract language from field name
            break

    # Get output
    output_text = data_point.get("output", "")

    # Get instruction language from instruct_lang if available, otherwise detect from content
    instruct_lang = data_point.get("instruct_lang", detect_instruction_language(instruction))

    # Store both instruction and input languages in metadata
    if "metadata" not in data_point:
        data_point["metadata"] = {}
    data_point["metadata"]["instruct_lang"] = instruct_lang
    data_point["metadata"]["input_lang"] = input_lang

    # Generate the full prompt using the instruction language
    full_prompt = prompter.generate_prompt(
        instruction,
        input_text,
        output_text,
        lang=instruct_lang
    )

    # Tokenize the prompt
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=config.cutoff_len,
        padding="max_length",
        add_special_tokens=True,
        return_tensors=None,
    )

    # Keep language metadata for evaluation
    if "metadata" in data_point:
        tokenized["metadata"] = data_point["metadata"]

    return tokenized


def compute_metrics(eval_pred, tokenizer=None):
    """
    Compute metrics using IdiomEvaluator for both tasks.
    
    This function serves as an interface between the Trainer and IdiomEvaluator,
    handling conversion of predictions and labels.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
        tokenizer: Tokenizer for decoding predictions
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Get predictions and labels
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # If no tokenizer is provided, raise an error (we should always pass it)
    if tokenizer is None:
        raise ValueError(
            "No tokenizer provided to compute_metrics. This function should be called with tokenizer passed."
        )

    # If predictions are logits, convert to token IDs
    if predictions.ndim == 3:  # Shape (batch_size, seq_length, vocab_size)
        predictions = np.argmax(predictions, axis=-1)

    # Replace padding tokens (-100) with pad_token_id
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)


    # Log the raw encoded tensors for debugging
    print("Sample encoded prediction (first 20 tokens):", predictions[0, :200])
    print("Sample encoded label (first 20 tokens):", labels[0, :200])

    # Clip values to valid token ID range to prevent errors
    max_vocab_id = tokenizer.vocab_size - 1
    predictions = np.clip(predictions, 0, max_vocab_id)
    labels = np.clip(labels, 0, max_vocab_id)

    # Ensure int64 dtype
    predictions = predictions.astype(np.int64)
    labels = labels.astype(np.int64)


    try:
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Log sample to help with debugging
        print("Sample prediction:", decoded_preds[0])
        print("Sample label:", decoded_labels[0][:200])

        # Initialize evaluator
        evaluator = IdiomEvaluator()

        # Process batch for Task 2
        processed_batch = evaluator.process_batch(decoded_labels, decoded_preds)

        # Compute metrics for both tasks
        task1_metrics = evaluator.compute_metrics_task1(decoded_preds, decoded_labels)
        task2_metrics = evaluator.compute_metrics_task2(processed_batch)

        metrics = {
            "task1_accuracy": task1_metrics["accuracy"],
            "task1_precision": task1_metrics["precision"],
            "task1_recall": task1_metrics["recall"],
            "task1_macro_f1": task1_metrics["macro_f1"],
            "task2_precision": task2_metrics["precision"],
            "task2_recall": task2_metrics["recall"],
            "task2_macro_f1": task2_metrics["macro_f1"]
        }
        
        # Log the computed metrics for both tasks
        logging.info("Task 1 Metrics: Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f", 
                task1_metrics["accuracy"], task1_metrics["precision"], task1_metrics["recall"], task1_metrics["macro_f1"])
        logging.info("Task 2 Metrics: Precision: %.4f, Recall: %.4f, F1: %.4f", 
                 task2_metrics["precision"], task2_metrics["recall"], task2_metrics["macro_f1"])

        return metrics

    except Exception as e:
        logging.error(f"Error in compute_metrics: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # Return minimum metrics to avoid training failure
        return {
            "macro_f1_class": 0,
            "macro_precision_class": 0,
            "macro_recall_class": 0,
            "precision": 0,
            "recall": 0,
            "macro_f1_char": 0,
            "macro_f1": 0
        }
def prepare_training_args(config: TrainingConfig) -> TrainingArguments:
    """
    Prepare training arguments based on the configuration.
    
    Sets up the HuggingFace Trainer arguments with appropriate parameters
    for effective training and evaluation.
    
    Args:
        config (TrainingConfig): Configuration object with parameters
        
    Returns:
        TrainingArguments: Configured arguments for the Trainer
    """
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
        learning_rate=3e-4,  # Initial learning rate
        weight_decay=0.01,  # Weight decay for regularization
        warmup_ratio=0.1,  # Percentage of steps for learning rate warmup
        logging_dir=os.path.join(config.output_dir, 'logs'),
        logging_strategy="steps",
        logging_steps=10,  # Log every 10 steps
        save_strategy="steps",
        save_steps=10,  # Save checkpoint every 10 steps
        evaluation_strategy="steps",
        eval_steps=10,  # Evaluate every 10 steps
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="eval_task2_macro_f1",  # Use Task 2 F1 score for best model
        greater_is_better=True,  # Higher F1 score is better
        fp16=True,  # Use mixed precision training
        gradient_checkpointing=True,  # Use gradient checkpointing for memory efficiency
        optim="paged_adamw_32bit",  # Use memory-efficient optimizer
        lr_scheduler_type="cosine_with_restarts",  # Cosine learning rate schedule with restarts
        max_grad_norm=1.0,  # Clip gradients to prevent explosion
        report_to="none",  # Don't report to external tracking services
        save_total_limit=4  # Keep only the 4 most recent checkpoints
    )

def generate_cross_language_analysis(results, output_dir):
    """
    Generate detailed cross-language analysis TSVs from evaluation results.
    
    Creates a set of TSV files for each metric, showing performance across
    different instruction and input language combinations.
    
    Args:
        results (dict): Evaluation results with cross-language data
        output_dir (str): Directory to save the analysis files
    """
    import os
    import json
    import csv
    import numpy as np
    from collections import defaultdict

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have cross-language data
    if 'test' not in results or 'by_instruction_input' not in results['test']:
        logging.error("Error: Results do not contain cross-language analysis data")
        return

    cross_lang_metrics = results['test']['by_instruction_input']

    # Extract unique instruction and input languages
    instruction_langs = set()
    input_langs = set()

    for lang_pair in cross_lang_metrics.keys():
        if "_" in lang_pair:
            instruct_lang, input_lang = lang_pair.split("_", 1)
            instruction_langs.add(instruct_lang)
            input_langs.add(input_lang)

    # Create sorted lists
    instruction_langs = sorted(list(instruction_langs))
    input_langs = sorted(list(input_langs))

    # Metrics to generate matrices for Task 1 and Task 2
    task1_metrics = [
        ("task1", "accuracy_task1", "Task 1 Accuracy"),
        ("task1", "precision_task1", "Task 1 Precision"),
        ("task1", "recall_task1", "Task 1 Recall"),
        ("task1", "macro_f1_task1", "Task 1 F1 Score")
    ]

    task2_metrics = [
        ("task2", "macro_precision", "Task 2 Precision"),
        ("task2", "macro_recall", "Task 2 Recall"),
        ("task2", "macro_f1", "Task 2 F1 Score")
    ]

    all_metrics = task1_metrics + task2_metrics

    # Generate matrix for each metric
    for task, metric_key, metric_name in all_metrics:
        # Create a matrix with instruction languages as rows and input languages as columns
        matrix = []
        # Add header row
        header_row = ["Instruction\\Input"] + input_langs
        matrix.append(header_row)

        # Add data rows
        for instruct_lang in instruction_langs:
            row = [instruct_lang]  # First column is the instruction language

            for input_lang in input_langs:
                lang_pair = f"{instruct_lang}_{input_lang}"

                if lang_pair in cross_lang_metrics and task in cross_lang_metrics[lang_pair]:
                    value = cross_lang_metrics[lang_pair][task].get(metric_key, 0)
                    row.append(f"{value:.4f}")
                    logging.info(f"Language Pair: {instruct_lang} -> {input_lang}, {metric_name}: {value:.4f}")
                else:
                    row.append("N/A")

            matrix.append(row)

        # Save to TSV
        output_file = os.path.join(output_dir, f"{metric_key}_matrix.tsv")
        with open(output_file, 'w', newline='', encoding='utf-8') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            writer.writerows(matrix)

        logging.info(f"Generated {metric_name} matrix at {output_file}")


def save_results(results: dict, output_dir: str):
    """
    Save evaluation results with enhanced cross-language analysis.
    
    Creates JSON files with detailed results and generates cross-language
    analysis files for easy visualization.
    
    Args:
        results (dict): Evaluation results with nested metrics
        output_dir (str): Directory to save the results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if the necessary keys exist
    if "test" not in results:
        logging.warning("The results dictionary does not contain a 'test' key.")
        return

    # Save detailed results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Create analysis directory
    analysis_dir = os.path.join(output_dir, 'cross_language_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # Generate cross-language analysis
    generate_cross_language_analysis(results, analysis_dir)

    # Save a traditional summary with language-specific metrics
    try:
        if 'test' in results and 'by_instruction' in results['test']:
            by_lang = results['test']['by_instruction']

            summary = {
                'test': {
                    lang: {
                        'task1': {
                            'macro_f1': by_lang[lang]['task1'].get('macro_f1_task1', 0),
                            'precision': by_lang[lang]['task1'].get('precision_task1', 0),
                            'recall': by_lang[lang]['task1'].get('recall_task1', 0)
                        },
                        'task2': {
                            'macro_f1': by_lang[lang]['task2'].get('macro_f1', 0),
                            'precision': by_lang[lang]['task2'].get('macro_precision', 0),
                            'recall': by_lang[lang]['task2'].get('macro_recall', 0)
                        }
                    } for lang in by_lang.keys() if lang != "overall"
                }
            }

            # Log Task 1 and Task 2 metrics for each language combination
            for lang, metrics in summary['test'].items():
                logging.info(f"Results for {lang}:")
                logging.info(f"  Task 1: F1: {metrics['task1']['macro_f1']}, Precision: {metrics['task1']['precision']}, Recall: {metrics['task1']['recall']}")
                logging.info(f"  Task 2: F1: {metrics['task2']['macro_f1']}, Precision: {metrics['task2']['precision']}, Recall: {metrics['task2']['recall']}")

            # Add overall results if available
            if 'overall' in by_lang:
                summary['test']['overall'] = {
                    'task1': {
                        'macro_f1': by_lang['overall']['task1'].get('macro_f1_task1', 0),
                        'precision': by_lang['overall']['task1'].get('precision_task1', 0),
                        'recall': by_lang['overall']['task1'].get('recall_task1', 0)
                    },
                    'task2': {
                        'macro_f1': by_lang['overall']['task2'].get('macro_f1', 0),
                        'precision': by_lang['overall']['task2'].get('macro_precision', 0),
                        'recall': by_lang['overall']['task2'].get('macro_recall', 0)
                    }
                }

            summary_file = os.path.join(output_dir, 'results_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"Error creating summary: {str(e)}")

def create_balanced_subset(dataset, fraction, balance_keys=None):
    """
    Creates a balanced subset of the dataset based on specified balance criteria.
    
    This function ensures that the subset is balanced across different categories
    like instruction language and presence of idioms, preventing skewed distributions.
    
    Args:
        dataset: The HuggingFace dataset to subset
        fraction: Fraction of the original dataset to keep (e.g., 0.032)
        balance_keys: List of balance criteria, each containing:
            - 'field': Field name to balance on
            - 'values': Optional list of specific values to balance
            - 'conditions': Optional list of lambda functions to categorize examples
            - 'labels': Optional labels corresponding to conditions
            
    Returns:
        A balanced subset of the original dataset
    """
    import numpy as np
    from collections import defaultdict
    
    # Convert dataset to list for easier manipulation
    dataset_list = dataset.to_list()
    total_subset_size = int(len(dataset_list) * fraction)
    
    if balance_keys is None:
        # Default balance criteria for idiom dataset
        balance_keys = [
            # Balance by instruction language
            {
                'field': 'instruct_lang',
                'values': ['en', 'it', 'pt']
            },
            # Balance by idiom presence (literal vs. idiomatic)
            {
                'field': 'output',
                'conditions': [
                    lambda x: x is None or x.lower() in ['none', 'nessuna', 'nenhuma'],
                    lambda x: x is not None and x.lower() not in ['none', 'nessuna', 'nenhuma']
                ],
                'labels': ['literal', 'idiomatic']
            }
        ]
    
    # Group examples based on all balance criteria
    groups = defaultdict(list)
    
    for idx, example in enumerate(dataset_list):
        # Create a group key based on all balance criteria
        key_parts = []
        
        for criterion in balance_keys:
            field = criterion['field']
            
            if 'values' in criterion and criterion['values']:
                # If specific values are provided
                value = example.get(field, "unknown")
                if value in criterion['values']:
                    key_parts.append(f"{field}_{value}")
                else:
                    key_parts.append(f"{field}_other")
            
            elif 'conditions' in criterion and criterion['conditions']:
                # If conditions are provided
                value = example.get(field)
                for i, condition in enumerate(criterion['conditions']):
                    if condition(value):
                        label = criterion.get('labels', [f"condition_{i}"])[i] if i < len(criterion.get('labels', [])) else f"condition_{i}"
                        key_parts.append(f"{field}_{label}")
                        break
                else:
                    # If no condition matches
                    key_parts.append(f"{field}_other")
            
            else:
                # If no values or conditions, use the field value directly
                key_parts.append(f"{field}_{example.get(field, 'unknown')}")
        
        # Join all parts to create a unique group key
        group_key = "_".join(key_parts)
        groups[group_key].append(idx)
    
    # Print statistics about the groups
    print(f"Found {len(groups)} unique example groups:")
    for key, indices in groups.items():
        print(f"  {key}: {len(indices)} examples")
    
    # Calculate how many examples to take from each group
    total_groups = len(groups)
    base_size = total_subset_size // total_groups
    remainder = total_subset_size % total_groups
    
    # Distribute the remainder across groups
    group_sizes = {key: base_size + (1 if i < remainder else 0) 
                  for i, key in enumerate(groups.keys())}
    
    # Adjust for groups that don't have enough examples
    deficit = 0
    for key, indices in groups.items():
        if len(indices) < group_sizes[key]:
            deficit += group_sizes[key] - len(indices)
            group_sizes[key] = len(indices)
    
    # Redistribute the deficit to other groups
    if deficit > 0:
        for key, indices in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
            available = len(indices) - group_sizes[key]
            if available > 0:
                to_add = min(deficit, available)
                group_sizes[key] += to_add
                deficit -= to_add
                if deficit == 0:
                    break
    
    # Sample from each group
    selected_indices = []
    for key, indices in groups.items():
        # Shuffle the indices for randomness
        np.random.shuffle(indices)
        # Take the required number of examples
        selected_indices.extend(indices[:group_sizes[key]])
    
    # Print final statistics
    print(f"Created balanced subset with {len(selected_indices)} examples:")
    for key in groups.keys():
        count = sum(1 for idx in selected_indices if idx in groups[key])
        print(f"  {key}: {count} examples ({count/len(selected_indices)*100:.1f}%)")
    
    # Return the subset using dataset.select()
    return dataset.select(selected_indices)


def main():
    """
    Main function to execute the training and evaluation process.
    
    This function orchestrates the entire training pipeline:
    1. Initialize configuration and tokenizer
    2. Load and process datasets
    3. Initialize the model
    4. Configure and run training
    5. Evaluate the model
    6. Save results
    
    Returns:
        dict: Evaluation results
    """
    # Initialize configuration
    config = TrainingConfig()

    # Initialize tokenizer with proper configuration
    tokenizer = initialize_tokenizer(config.base_model)

    # Initialize prompter after tokenizer
    prompter = Prompter(config.prompt_template_name)

    # Load datasets
    train_val = load_dataset("json", data_files={"train": "/home/dciminari/thesis/data/merged_train.json"})
    test_dataset = load_dataset("json", data_files={"test": "/home/dciminari/thesis/data/merged_test.json"})

    # Split datasets
    train_val_split = train_val["train"].train_test_split(
        test_size=config.val_set_size,
        shuffle=True,
        seed=42
    )

    # Define balance criteria
    balance_criteria = [
        # Balance by instruction language
        {
            'field': 'instruct_lang',
            'values': ['en', 'it', 'pt']
        },
        # Balance by idiom presence (literal vs. idiomatic)
        {
            'field': 'output',
            'conditions': [
                lambda x: x is None or (isinstance(x, str) and x.lower() in ['none', 'nessuna', 'nenhuma']),
                lambda x: x is not None and (not isinstance(x, str) or x.lower() not in ['none', 'nessuna', 'nenhuma'])
            ],
            'labels': ['literal', 'idiomatic']
        }
    ]

    # Create balanced subsets
    train_subset = create_balanced_subset(
        train_val_split["train"], 
        config.train_subset_fraction,
        balance_criteria
    )

    val_subset = create_balanced_subset(
        train_val_split["test"], 
        config.val_subset_fraction,
        balance_criteria
    )

    test_subset = create_balanced_subset(
        test_dataset["test"], 
        config.test_subset_fraction,
        balance_criteria
    )

    logging.info(f"Created balanced subsets - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")
    
    # Process datasets
    train_cols_to_remove = [
        col for col in train_subset.column_names
        if col not in ["metadata"] and not (col == "instruction" or col.startswith("input_"))
    ]

    val_cols_to_remove = [
        col for col in val_subset.column_names
        if col not in ["metadata"] and not (col == "instruction" or col.startswith("input_"))
    ]

    # Tokenize and prepare datasets
    train_data = train_subset.map(
        lambda x: generate_and_tokenize_prompt(x, prompter, tokenizer, config, is_training=True),
        remove_columns=train_cols_to_remove
    )

    val_data = val_subset.map(
        lambda x: generate_and_tokenize_prompt(x, prompter, tokenizer, config, is_training=False),
        remove_columns=val_cols_to_remove
    )

    # Test data remains the same
    test_cols_to_remove = [
        col for col in test_dataset["test"].column_names
        if col not in ["metadata"] and not (col == "instruction" or col.startswith("input_"))
    ]

    test_data = test_subset.map(
        lambda x: generate_and_tokenize_prompt(x, prompter, tokenizer, config, is_training=False),
        remove_columns=test_cols_to_remove
    )

    # Initialize model with checkpoint loading
    model = initialize_model(config, tokenizer)

    # Prepare training arguments
    training_args = prepare_training_args(config)

    # Modify training arguments for checkpoint resumption
    training_args.resume_from_checkpoint = config.resume_from_checkpoint

    # Configure logging more verbosely
    logging.info("=" * 80)
    logging.info("TRAINING SETUP")
    logging.info("=" * 80)
    logging.info(f"Model: {config.base_model}")
    logging.info(f"Training subset fraction: {config.train_subset_fraction} ({len(train_data)} samples)")
    logging.info(f"Validation subset fraction: {config.val_subset_fraction} ({len(val_data)} samples)")
    logging.info(f"Test subset fraction: {config.test_subset_fraction} ({len(test_data)} samples)")
    logging.info(f"Language instruction detection: enabled")
    logging.info(f"Cross-language metrics: enabled")
    logging.info("=" * 80)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize trainer
    trainer = CustomTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train with checkpoint resumption
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Set up final evaluation directory
    final_eval_dir = os.path.join(config.output_dir, 'final_evaluation')
    os.makedirs(final_eval_dir, exist_ok=True)

    logging.info("=" * 80)
    logging.info("FINAL EVALUATION")
    logging.info("=" * 80)
    logging.info(f"Evaluating on {len(test_data)} test samples")
    
    # Evaluate with all language combinations logged
    evaluator = IdiomEvaluator()
    output = trainer.predict(test_data)
    predictions, labels = output.predictions, output.label_ids
    decoded_preds, decoded_labels = trainer._decode_predictions_labels(predictions, labels)
    
    # Evaluate with detailed language breakdown, this calls our modified methods
    results = evaluator.evaluate_by_language(
        decoded_preds, 
        decoded_labels,
        test_data=test_data,
        val_data=val_data,
        output_dir=final_eval_dir
    )

    # Save final results
    save_results(results, final_eval_dir)
    
    logging.info("Final evaluation complete and results saved to %s", final_eval_dir)
    logging.info("=" * 80)

    return results

if __name__ == "__main__":
    main()
