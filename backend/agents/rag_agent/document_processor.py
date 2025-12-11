import os 
import logging 
from typing import List, Dict, Any 
import pypdf
from docx import Document
from bs4 import BeautifulSoup 
import tiktoken

from config import config, logger

class DocumentProcessor:
    """Handles document parsing and text extraction"""

    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info("DocumentProcessor initialized")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""

    def extract_text_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            return ""

    def extract_text_from_html(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file.read(), "html.parser")
                return soup.get_text().strip()
        except Exception as e:
            logger.error(f"Error extracting text from HTML {file_path}: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_extension = file_path.split('.')[-1].lower()

        if file_extension == "pdf":
            return self.extract_text_from_pdf(file_path)
        elif file_extension == "docx":
            return self.extract_text_from_docx(file_path)
        elif file_extension == "txt":
            return self.extract_text_from_txt(file_path)
        elif file_extension in ["html", "htm"]:
            return self.extract_text_from_html(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Split text into chunks with overlap"""
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = config.CHUNK_OVERLAP
        
        # Tokenize the text
        tokens = self.encoding.encode(text)

        chunks = []
        start = 0

        while start < len(tokens): 
            end = start + chunk_size 
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(chunk_text)

            # Move start position with overlap
            start = end - chunk_overlap
            
            # Prevent infinite loop
            if start >= len(tokens):
                break
                
        logger.info(f"Text chunked into {len(chunks)} pieces")
        return chunks

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and return structured data"""
        try:
            # Extract text
            text = self.extract_text(file_path)
            if not text:
                return {"success": False, "message": "No text extracted"}

            # Chunk the text
            chunks = self.chunk_text(text)

            # Create document metadata
            doc_id = os.path.basename(file_path)
            doc_metadata = {
                "doc_id": doc_id,
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "chunk_account": len(chunks),
                "total_tokens": len(self.encoding.encode(text))
            }

            return {
                "success": True,
                "text": text,
                "chunks": chunks,
                "metadata": doc_metadata
            }

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {"success": False, "error": str(e)}

# Global document processor instance
document_processor = DocumentProcessor()
