from typing import List
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain.schema import Document

class ChunkingStrategy:
    """
    Challenge #2: Finding optimal chunk size and overlap
    Interview answer: "I experimented with different chunking strategies:
    - Started with 500 tokens, found answers were incomplete
    - Increased to 1000 tokens but had context bleeding issues
    - Settled on 800 tokens with 200 overlap for best balance
    
    Measured using retrieval accuracy and answer completeness metrics"
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def recursive_character_split(self, documents: List[Document]) -> List[Document]:
        """
        Best for most use cases - respects natural text boundaries
        This is what I recommend starting with
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk metadata for tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
    
    def token_based_split(self, documents: List[Document]) -> List[Document]:
        """
        Token-based splitting - useful when you need precise token control
        Use this when working with token limits
        """
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_documents(documents)
    
    def semantic_split(self, documents: List[Document]) -> List[Document]:
        """
        Advanced: Split by semantic meaning (sentences/paragraphs)
        For interview stories, you might want to keep complete stories together
        """
        text_splitter = CharacterTextSplitter(
            separator="\n\n",  # Split by paragraphs
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    def compare_strategies(self, documents: List[Document]) -> dict:
        """
        Compare different chunking approaches
        Returns statistics for each strategy
        """
        strategies = {
            'recursive': self.recursive_character_split(documents),
            'token': self.token_based_split(documents),
            'semantic': self.semantic_split(documents),
        }
        
        stats = {}
        for name, chunks in strategies.items():
            stats[name] = {
                'num_chunks': len(chunks),
                'avg_chunk_size': sum(len(c.page_content) for c in chunks) / len(chunks),
                'min_chunk_size': min(len(c.page_content) for c in chunks),
                'max_chunk_size': max(len(c.page_content) for c in chunks),
            }
        
        return stats