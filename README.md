# MultIdiomLlama
## Overview
This repository contains the scripts used for the thesis "A Tough Row to Hoe: Instruction Fine-Tuning LLaMA 3.2 for Multilingual Sentence Disambiguation and Idiom Identification", which explores multilingual automatic idiom processing using LLaMA 3.2 1B. Specifically, this thesis addresses two tasks: sentence disambiguation and idiom identification. Sentence disambiguation is framed as a binary classification problem, where a sentence can be classified as either literal or idiomatic. Idiom identification is a span identification problem, where the model is tasked with identifying the span within the sentence that contains the idiomatic expressions. 
To tackle these tasks, I created a multilingual dataset consisting of instruction-formatted samples in English, Italian, and Portuguese. Each sample includes an instruction (the task description), an input sentence, and the expected output. To create this dataset, I used three already annotated datasets: [ID10M](https://github.com/Babelscape/ID10M), [AStitchInLanguageModels](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels), and MultiCoPIE (to be released).
Then, based on this dataset, I adopted two distinct approaches, i.e. prompting and instruction fine-tuning, to investigate the impact of the instruction language on the model's performance.

## Content
This repository contains three subfolders:
1. `fine_tuning`: this subfolder contains the scripts to fine-tune [LLaMA 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) on the instruction dataset;
2. `prompting`: this subfolder includes the code to prompt [LLaMA 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) in a zero-shot and a few-shot setting;
3. `data`: this subfolder contains training and test data in English, Italian, and Portuguese.

## Requirements
To install the required dependencies, the `requirements.txt` file can be used as follows: 

```bash
pip install -r requirements.txt
```
## License
The MultIdiomLlama dataset is licensed under the CC BY-SA-NC 4.0 license.

<sub>The dataset is derived from three corpora with different licenses; the most restrictive license was applied to this project’s dataset.</sub>
