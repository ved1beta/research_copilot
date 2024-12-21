import PyPDF2
import re
from typing import Dict, Any
import os
import pdfplumber


class PDFExtractor:
    def __init__(self, storage_path: str = "../../../data/processed"):
        self.storage_path = storage_path
        self.section_patterns = [
            # Common section title patterns
            r'^(?:\d+\.)?\s*abstract\s*$',
            r'^(?:\d+\.)?\s*introduction\s*$',
            r'^(?:\d+\.)?\s*(?:methodology|methods|materials and methods)\s*$',
            r'^(?:\d+\.)?\s*(?:results|findings)\s*$',
            r'^(?:\d+\.)?\s*discussion\s*$',
            r'^(?:\d+\.)?\s*conclusion(?:s)?\s*$',
            r'^(?:\d+\.)?\s*references\s*$',
            r'^(?:\d+\.)?\s*background\s*$',
            r'^(?:\d+\.)?\s*related work\s*$',
            r'^(?:\d+\.)?\s*experimental setup\s*$',
            r'^(?:\d+\.)?\s*evaluation\s*$',
            r'^[A-Z][A-Z ]+$',  # All-uppercase titles (e.g., INTRODUCTION)
        ]
        self.section_patterns = [re.compile(p, re.IGNORECASE) for p in self.section_patterns]
        os.makedirs(storage_path, exist_ok=True)

    def is_section_header(self, line: str) -> bool:
        """
        Determine if a line is likely a section header based on multiple criteria
        """
        # Skip empty lines
        if not line.strip():
            return False

        line = line.strip().lower()

        # Check against section patterns
        return any(pattern.match(line) for pattern in self.section_patterns)

    def clean_section_name(self, header: str) -> str:
        """Clean and standardize section names"""
        # Remove numbers and extra spaces
        clean = re.sub(r'^\d+\.?\s*', '', header)
        clean = clean.strip().lower()

        # Map similar section names to standard ones
        section_mapping = {
            'methodology': 'methods',
            'materials and methods': 'methods',
            'experimental setup': 'methods',
            'findings': 'results',
            'conclusions': 'conclusion'
        }

        return section_mapping.get(clean, clean)

    def replace_ligatures(self, text: str) -> str:
        """Replace common ligatures and special characters"""
        ligature_map = {
            "\ufb01": "fi",
            "\ufb02": "fl",
            "\u201c": "\"",
            "\u201d": "\"",
            "\u2013": "-",
            "\u2014": "--"
        }
        for ligature, replacement in ligature_map.items():
            text = text.replace(ligature, replacement)
        return text

    def extract_text_with_sections(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF with improved section detection"""
        text_content = {
            "metadata": {
                "title": "",
                "sections_found": [],
                "total_pages": 0
            },
            "sections": {},
            "full_text": ""
        }

        current_section = "unknown"
        current_text = []
        all_text = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_content["metadata"]["total_pages"] = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text is None:
                        continue
                    text = self.replace_ligatures(text)
                    lines = text.split('\n')

                    for line in lines:
                        if not line.strip():
                            continue

                        # Check if this line is a section header
                        if self.is_section_header(line):
                            # Save the previous section
                            if current_text:
                                section_text = '\n'.join(current_text).strip()
                                if section_text:
                                    text_content["sections"][current_section] = section_text

                            # Start new section
                            current_section = self.clean_section_name(line.strip())
                            if current_section not in text_content["metadata"]["sections_found"]:
                                text_content["metadata"]["sections_found"].append(current_section)
                            current_text = []
                        else:
                            current_text.append(line.strip())
                            all_text.append(line.strip())

                if current_text:
                    section_text = '\n'.join(current_text).strip()
                    if section_text:
                        text_content["sections"][current_section] = section_text

                text_content["full_text"] = '\n'.join(all_text)

                # Attempt to extract title from the first page
                first_page_text = pdf.pages[0].extract_text()
                first_lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
                if first_lines:
                    text_content["metadata"]["title"] = " ".join(first_lines[:3])

        except Exception as e:
            text_content["metadata"]["error"] = str(e)

        return text_content
