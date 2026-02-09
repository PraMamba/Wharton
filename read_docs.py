import zipfile
import os
import PyPDF2
from docx import Document

def extract_and_read(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract specific files
        files_to_extract = [
            'WHSDSC 2026 Workbook - Hockey.pdf',
            'WHSDSC 2026 Glossary.docx'
        ]
        
        # Find actual paths in zip (handle potential subdirectories)
        target_files = {}
        for file in zip_ref.namelist():
            for target in files_to_extract:
                if file.endswith(target):
                    target_files[target] = file
        
        if not target_files:
            print("Target documentation files not found in zip.")
            return

        zip_ref.extractall(path='temp_docs', members=target_files.values())
        
        # Read PDF
        if 'WHSDSC 2026 Workbook - Hockey.pdf' in target_files:
            pdf_path = os.path.join('temp_docs', target_files['WHSDSC 2026 Workbook - Hockey.pdf'])
            print(f"\n--- Reading {pdf_path} ---")
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    # Read first few pages which usually contain the prompt/challenge
                    num_pages = min(len(reader.pages), 5)
                    for i in range(num_pages):
                        text = reader.pages[i].extract_text()
                        print(f"\nPage {i+1}:\n{text}")
            except Exception as e:
                print(f"Error reading PDF: {e}")

        # Read DOCX
        if 'WHSDSC 2026 Glossary.docx' in target_files:
            docx_path = os.path.join('temp_docs', target_files['WHSDSC 2026 Glossary.docx'])
            print(f"\n--- Reading {docx_path} ---")
            try:
                doc = Document(docx_path)
                print("\nDocument Content:")
                for para in doc.paragraphs[:20]: # Read first 20 paragraphs
                    if para.text.strip():
                        print(para.text)
            except Exception as e:
                print(f"Error reading DOCX: {e}")

if __name__ == "__main__":
    extract_and_read('drive-download-20260205T132825Z-1-001.zip')
