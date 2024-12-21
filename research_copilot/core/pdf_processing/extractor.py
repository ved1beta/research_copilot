import PyPDF2
import re
from typing import Dict, Any
import os


class PDFExtractor:
    def __init__(self, storage_path: str = "../../../data/processed"):
        self.storage_path = storage_path
        self.section_patterns = [
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
            r'^(?:\d+\.)?\s*evaluation\s*$'
        ]
        self.section_patterns = [re.compile(p, re.IGNORECASE) for p in self.section_patterns]
        os.makedirs(storage_path, exist_ok=True)

    def is_section_header(self, line: str) -> bool:
        if not line.strip():
            return False
        line = line.strip().lower()
        return any(pattern.match(line) for pattern in self.section_patterns)

    def clean_section_name(self, header: str) -> str:
        clean = re.sub(r'^\d+\.?\s*', '', header)
        clean = clean.strip().lower()
        section_mapping = {
            'methodology': 'methods',
            'materials and methods': 'methods',
            'experimental setup': 'methods',
            'findings': 'results',
            'conclusions': 'conclusion'
        }
        return section_mapping.get(clean, clean)

    def extract_text_with_sections(self, pdf_path: str) -> Dict[str, Any]:
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
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                text_content["metadata"]["total_pages"] = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    lines = text.split('\n')

                    for line in lines:
                        if not line.strip():
                            continue

                        if self.is_section_header(line):
                            if current_text:
                                section_text = '\n'.join(current_text).strip()
                                if section_text:
                                    text_content["sections"][current_section] = section_text
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

                first_page_text = pdf.pages[0].extract_text()
                first_lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
                if first_lines:
                    text_content["metadata"]["title"] = first_lines[0]

        except Exception as e:
            text_content["metadata"]["error"] = str(e)

        return text_content
