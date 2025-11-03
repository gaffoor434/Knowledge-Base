import unittest
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/app/backend/services')

from backend.services.document_processor import process_document
from backend.services.query_engine import query_knowledge_base
from backend.services.bm25_service import BM25Service
from backend.services.qdrant_service import get_qdrant_client, QDRANT_COLLECTION

class TestBackendPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        from backend.services.qdrant_service import ensure_collection_exists
        cls.qdrant_client = get_qdrant_client()
        ensure_collection_exists()
        cls.bm25_service = BM25Service(index_path="test_bm25_index.pkl")

        # Define test file paths
        cls.test_files_dir = os.path.join(os.path.dirname(__file__), "test_files")
        os.makedirs(cls.test_files_dir, exist_ok=True)

        cls.excel_path = os.path.join(cls.test_files_dir, "test_excel.xlsx")
        cls.pdf_path = os.path.join(cls.test_files_dir, "test_mom.pdf")

        # Create dummy files for testing
        cls.create_dummy_files()

        # Process the documents
        process_document(cls.excel_path)
        process_document(cls.pdf_path)

        # Rebuild the BM25 index with the new documents
        from rebuild_bm25_index import fetch_all_documents_from_qdrant
        all_docs = fetch_all_documents_from_qdrant()
        cls.bm25_service.build_index(all_docs)

    @classmethod
    def create_dummy_files(cls):
        """Creates dummy Excel and PDF files for testing."""
        import pandas as pd
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        # Create a dummy Excel file
        df = pd.DataFrame({
            "Name": ["Alice", "Bob"],
            "Age": [30, 25],
            "City": ["New York", "Los Angeles"]
        })
        df.to_excel(cls.excel_path, index=False, sheet_name="Sheet1")

        # Create a dummy PDF with a table and action items
        c = canvas.Canvas(cls.pdf_path, pagesize=letter)
        c.drawString(100, 750, "Sprint Planning Session")
        c.drawString(100, 700, "Action Items:")
        c.drawString(120, 680, "- Alice to finalize the report.")
        c.drawString(120, 660, "- Bob to review the code.")

        # Add a simple table
        c.drawString(100, 600, "Team Roster")
        c.drawString(100, 580, "Name | Role")
        c.drawString(100, 560, "Alice | Developer")
        c.drawString(100, 540, "Bob | QA")
        c.save()

    def test_01_excel_table_extraction(self):
        """Verify that data from the Excel sheet is correctly extracted and searchable."""
        answer, sources = query_knowledge_base("What is Alice's age?", self.bm25_service)
        self.assertIn("30", answer)
        self.assertIn("test_excel.xlsx", sources)

    def test_02_pdf_parsing_and_action_items(self):
        """Verify that action items are correctly parsed from the PDF."""
        answer, sources = query_knowledge_base("List all action items.", self.bm25_service)
        self.assertIn("Alice", answer)
        self.assertIn("Bob", answer)
        self.assertIn("test_mom.pdf", sources)

    def test_03_negative_case_i_dont_know(self):
        """Test that the system returns 'I don't know' for out-of-scope questions."""
        answer, sources = query_knowledge_base("What is the capital of France?", self.bm25_service)
        self.assertIn("I don't know", answer)
        self.assertEqual(len(sources), 0)

    def test_04_source_attribution(self):
        """Check that the response includes the correct source document."""
        answer, sources = query_knowledge_base("Who is the QA?", self.bm25_service)
        self.assertIn("Bob", answer)
        self.assertIn("test_mom.pdf", sources)
        self.assertNotIn("test_excel.xlsx", sources)

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        # Delete the test files
        os.remove(cls.excel_path)
        os.remove(cls.pdf_path)
        os.rmdir(cls.test_files_dir)

        # Clean up the test index
        if os.path.exists("test_bm25_index.pkl"):
            os.remove("test_bm25_index.pkl")

        # Optional: Clean up Qdrant collection if needed, be careful with this
        # cls.qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)

if __name__ == "__main__":
    unittest.main()
