import os 
import plumber
import re
class PDFExtractor:
    def __init__(self, storage_path: str = "data/processed"):
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
            r'^(?:\d+\.)?\s*evaluation\s*$'
        ]
        self.section_patterns = [re.compile(p, re.IGNORECASE) for p in self.section_patterns]
        os.makedirs(storage_path, exist_ok=True)

    def section_header(self, line:str, page_properties:dict) -> bool:
        if not line.strip():
            return False
        line.strip().lower()

        if page_properties:
            # Check font size
            font_size = page_properties.get('font_size')
            if font_size and font_size > 14:  # assuming larger font sizes indicate headings
                return True

            # Check font style (bold, italic, etc.)
            font_style = page_properties.get('font_style')
            if font_style and 'bold' in font_style:
                return True

        # Check against section patterns
        return any(pattern.match(line) for pattern in self.section_patterns)