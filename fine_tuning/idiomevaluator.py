import os
import csv
import numpy as np
import json
from collections import defaultdict
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class IdiomEvaluator:
    """    
    This class provides methods to evaluate model performance on two tasks:
    1. Task 1: Binary classification - detecting whether an idiom is present
    2. Task 2: Span identification - extracting the exact idiom text
    """

    def __init__(self):
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

    def compute_metrics_task1(self, decoded_preds, decoded_labels):
            """
            Compute Task 1 metrics.
            
            Task 1 is the binary classification task of detecting whether an idiom is present.
            This method calculates accuracy, precision, recall, and F1 score.
            
            Args:
                decoded_preds (list): List of decoded model predictions
                decoded_labels (list): List of decoded gold standard labels
                
            Returns:
                dict: Dictionary of metrics with standard deviations
            """
            self.logger.info(f"Computing Task 1 metrics for {len(decoded_preds)} samples")

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
                    "macro_f1_task1": 0
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
                # Return comprehensive metrics
                return {
                    "accuracy_task1": acc,
                    "precision_task1": precision,
                    "recall_task1": recall,
                    "macro_f1_task1": f1
                }

            except Exception as e:
                self.logger.error(f"Error computing Task 1 metrics: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {
                    "accuracy_task1": 0,
                    "precision_task1": 0,
                    "recall_task1": 0,
                    "macro_f1_task1": 0
                }


    def compute_metrics_task2(self, processed_batch):
        """
        Compute Task 2 metrics for idiom span identification.
        
        Task 2 evaluates how well the model extracts the exact idiom span.
        Only samples with idioms present are included in this evaluation.
        
        Args:
            processed_batch (dict): Batch of processed data with input_texts, labels, predictions, has_idiom
        Returns:
            dict: Dictionary of metrics with standard deviations
        """

        # Get data for samples with idioms only
        idiom_indices = [i for i, has_idiom in enumerate(processed_batch["has_idiom"]) if has_idiom == 1]

        if not idiom_indices:
            self.logger.warning("No samples with idioms found for Task 2 metrics")
            return {
                "macro_precision": 0,
                "macro_recall": 0,
                "macro_f1": 0,
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

            return {
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
                "individual_metrics": individual_data
            }

        except Exception as e:
            self.logger.error(f"Error computing Task 2 metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "macro_precision": 0,
                "macro_recall": 0,
                "macro_f1": 0
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
                    'value'
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
                            if lang not in "overall":
                                # Task 1 metrics
                                if "task1" in results_by_lang[lang]:
                                    task1 = results_by_lang[lang]["task1"]
                                    for metric_name, value in task1.items():
                                        if not metric_name.endswith("_std"):
                                            writer.writerow({
                                                'language': lang,
                                                'task': 'task1',
                                                'metric': metric_name,
                                                'value': f"{value:.4f}"
                                            })

                                # Task 2 metrics
                                if "task2" in results_by_lang[lang]:
                                    task2 = results_by_lang[lang]["task2"]
                                    for metric_name, value in task2.items():
                                        if metric_name != "individual_metrics" and not metric_name.endswith("_std"):

                                            writer.writerow({
                                                'language': lang,
                                                'task': 'task2',
                                                'metric': metric_name,
                                                'value': f"{value:.4f}",
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
                                                'value': f"{value:.4f}"
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