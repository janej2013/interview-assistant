from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain.schema import Document
import os

class DocumentLoader:
    """
    Challenge #1: Handling different document formats
    Interview answer: "I had to support multiple formats (txt, pdf, docx)
    and handle encoding issues with different text files"
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def load_text_files(self) -> List[Document]:
        """Load all text files from directory"""
        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True}
        )
        return loader.load()
    
    def load_pdf_files(self) -> List[Document]:
        """Load all PDF files"""
        docs = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(self.data_dir, file))
                docs.extend(loader.load())
        return docs
    
    def load_all_documents(self) -> List[Document]:
        """Load all supported document types"""
        all_docs = []
        
        # Try loading different formats
        try:
            all_docs.extend(self.load_text_files())
        except Exception as e:
            print(f"Error loading text files: {e}")
        
        try:
            all_docs.extend(self.load_pdf_files())
        except Exception as e:
            print(f"Error loading PDF files: {e}")
        
        # Add metadata
        for i, doc in enumerate(all_docs):
            doc.metadata['doc_id'] = i
            doc.metadata['source'] = doc.metadata.get('source', 'unknown')
        
        print(f"Loaded {len(all_docs)} documents")
        return all_docs
    
    def load_from_string(self, text: str, metadata: dict = None) -> Document:
        """Create document from string (useful for testing)"""
        return Document(
            page_content=text,
            metadata=metadata or {}
        )