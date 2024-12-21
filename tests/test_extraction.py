import sys
import os
import unittest
from typing import Any, Dict
import pdfplumber

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research_copilot.core.pdf_processing.extractor import PDFExtractor


class TestPDFExtractor(unittest.TestCase):
    def test_pdf_processing(self):
        # Initialize extractor
        extractor = PDFExtractor()

        # Test with a sample PDF
        pdf_path = "../data/uploads/1301.3781v3.pdf"
        if not os.path.exists(pdf_path):
            self.fail(f"Please place a test PDF at {pdf_path}")

        # Process the PDF
        print(f"Processing PDF at {pdf_path}")
        results: Dict[str, Any] = extractor.extract_text_with_sections(pdf_path)

        # Print results for debugging
        print("Extracted Data:")
        print(results)

        # Check results
        if 'error' in results["metadata"]:
            self.fail(f"An error occurred during extraction: {results['metadata']['error']}")
        else:
            self.assertIn('sections', results)
            self.assertIn('metadata', results)
            self.assertGreater(results['metadata']['total_pages'], 0)
            self.assertTrue(results['metadata']['sections_found'])


if __name__ == '__main__':
    unittest.main()
