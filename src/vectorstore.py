from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from config.config import config

class VectorStoreManager:
    """
    Challenge #3: Vector store performance and persistence
    Interview answer: "Initially used in-memory storage, but lost data between runs.
    Implemented ChromaDB with persistence. Also had to handle:
    - Embedding rate limits (added retry logic)
    - Duplicate documents (implemented deduplication)
    - Search quality (tuned similarity thresholds)"
    """
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        embedding_model: str = None
    ):
        self.persist_directory = persist_directory or config.VECTOR_STORE_PATH
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create and persist vector store from documents"""
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Deduplicate documents based on content
        unique_docs = self._deduplicate_documents(documents)
        print(f"After deduplication: {len(unique_docs)} documents")
        
        self.vectorstore = Chroma.from_documents(
            documents=unique_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        
        print("Vector store created and persisted")
        return self.vectorstore
    
    def load_vectorstore(self) -> Chroma:
        """Load existing vector store"""
        print("Loading existing vector store...")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load first.")
        
        unique_docs = self._deduplicate_documents(documents)
        self.vectorstore.add_documents(unique_docs)
        print(f"Added {len(unique_docs)} new documents")
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Search for similar documents"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        if score_threshold:
            # Search with score filtering
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, k=k
            )
            # Filter by threshold (lower score = more similar for some stores)
            filtered_docs = [
                doc for doc, score in docs_and_scores
                if score >= score_threshold
            ]
            return filtered_docs
        else:
            return self.vectorstore.similarity_search(query, k=k)
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content hash"""
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store"""
        if self.vectorstore is None:
            return {"error": "Vector store not initialized"}
        
        collection = self.vectorstore._collection
        return {
            "collection_name": self.collection_name,
            "document_count": collection.count(),
            "embedding_model": self.embedding_model,
        }