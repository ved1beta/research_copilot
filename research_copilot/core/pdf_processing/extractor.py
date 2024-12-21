from typing import Dict, Any

class PDFExtractor:
    def __init__(self):
        self.supported_formats = ['pdf']
    
    def extract_text(self, file_path: str) -> Dict[str, Any]:
        '''Extract text and metadata from PDF'''
        raise NotImplementedError
    
    def extract_figures(self, file_path: str) -> Dict[str, Any]:
        '''Extract figures and tables from PDF'''
        raise NotImplementedError