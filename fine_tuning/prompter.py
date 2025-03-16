"""
A dedicated helper to manage templates and prompt building with multilingual support.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("templates", "_verbose", "default_lang")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.default_lang = "en"  # Default language
        
        if not template_name:
            template_name = "alpaca"
            
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
            
        with open(file_name) as fp:
            templates = json.load(fp)
            
        # Convert the list of templates to a dictionary for easy access by language
        self.templates = {}
        if isinstance(templates, list):
            for template in templates:
                lang = template.get("lang", "en")
                self.templates[lang] = template
        else:
            # Handle the case where the template file is in the old format (single template)
            self.templates["en"] = templates
            
        if self._verbose:
            print(f"Using prompt template {template_name} with {len(self.templates)} language variants")

    def _detect_language(self, filename: str) -> str:
        """
        Detect language based on filename conventions only. 
        It can detect the language of a prompt based on filename prefixes (e.g., "EN_" for English, "IT_" for Italian, "PT_" for Portuguese).
        
        Args:
            filename: The name of the input file
        
        Returns:
            Language code: 'en', 'it', or 'pt'
        """
        if filename:
            filename_upper = filename.upper()
            
            if filename_upper.startswith('EN_'):
                return 'en'
            elif filename_upper.startswith('IT_'):
                return 'it'
            elif filename_upper.startswith('PT_'):
                return 'pt'
        
        # Fallback to default language if no match is found
        return self.default_lang

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
        lang: Union[None, str] = None,
        filename: Union[None, str] = None
    ) -> str:
        """
        Generate a properly formatted prompt using language from the data or filename.
        
        Args:
            instruction: The instruction text
            input: Optional input text
            label: Optional label/response text
            lang: Optional language code directly provided
            filename: Optional filename to detect language from
        
        Returns:
            Formatted prompt string
        """
        # Sanitize inputs
        instruction = instruction.strip() if instruction else ""
        input = input.strip() if input else ""
        label = label.strip() if label else ""
        
        # Determine language - prioritize directly provided language
        if lang and lang in self.templates:
            detected_lang = lang
        else:
            # Fall back to filename detection if no direct language is provided
            detected_lang = self._detect_language(filename) if filename else self.default_lang
        
        # Get the appropriate template for the detected language
        template = self.templates.get(detected_lang, self.templates.get(self.default_lang))
        
        try:
            # Use template with input
            prompt_template = template.get("prompt_input", "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
            res = prompt_template.format(
                instruction=instruction,
                input=input
            )            
            # Get response marker based on language
            response_marker = "### Response:"
            if detected_lang == "it":
                response_marker = "### Risposta:"
            elif detected_lang == "pt":
                if not res.endswith("\n"):
                    res += "\n"
                res += label
                
            # Normalize whitespace
            res = "\n".join(line.strip() for line in res.split("\n"))
            
            return res
        
        except Exception as e:
            print(f"Error generating prompt with language {detected_lang}: {str(e)}")
            # Fallback to basic format
            if input:
                fallback = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
            else:
                fallback = f"### Instruction:\n{instruction}\n\n"
                
            # Add appropriate response marker
            if detected_lang == "it":
                fallback += "### Risposta:\n"
            elif detected_lang == "pt":
                fallback += "### Resposta:\n"
            else:
                fallback += "### Response:\n"
                
            if label:
                fallback += label
                
            return fallback
