import os
import torch
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any

class BatchBGEEmbeddings(Embeddings):
    """BGE embedding function optimized for batch processing"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents for storage in vector database"""
        if not texts:
            return []
            
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            batch = texts[i:i + self.batch_size]
            
            # Add BGE prefix for documents
            prefixed_texts = [
                f"Represent this sentence for searching relevant passages: {text}" 
                for text in batch
            ]
            
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=len(batch),
                normalize_embeddings=True,
                convert_to_numpy=True,
                device='cuda' if self.model.device.type == 'cuda' else 'cpu'
            )
            
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text for similarity search"""
        query_text = f"Represent this sentence for searching relevant passages: {text}"
        
        embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            device='cuda' if self.model.device.type == 'cuda' else 'cpu'
        )
        
        return embedding[0].tolist()

class VectorDB:
    """Vector database wrapper using LangChain Chroma implementation"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", batch_size: int = 32):
        self.embeddings = BatchBGEEmbeddings(model_name=model_name, batch_size=batch_size)
        self.vector_store = None
        self.persist_directory = None
        
    def initialize_vector_store(
        self, 
        persist_directory: str, 
        collection_name: str = "movies_collection"
    ) -> str:
        """Initialize the LangChain Chroma vector store"""
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        
        # Initialize LangChain Chroma vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        current_count = self.count()
        return f"Vector store initialized at {persist_directory}. Current documents: {current_count}"
    
    def add_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 25,
        verbose: bool = True
    ) -> int:
        """Add documents to the vector store in batches"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize_vector_store() first.")
        
        total_docs = len(documents)
        if verbose:
            print(f"Adding {total_docs} documents in batches of {batch_size}")
        
        # Add documents in batches
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            if verbose:
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} docs)")
            
            # Add batch to vector store
            self.vector_store.add_documents(batch)
            
            if verbose:
                current_count = self.count()
                print(f"Batch {batch_num} completed. Total docs in DB: {current_count}")
        
        final_count = self.count()
        if verbose:
            print(f"All documents added successfully. Final count: {final_count}")
        
        return final_count
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform similarity search"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize_vector_store() first.")
        
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Perform similarity search with relevance scores"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize_vector_store() first.")
        
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def as_retriever(self, **kwargs):
        """Get LangChain retriever interface"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize_vector_store() first.")
        
        return self.vector_store.as_retriever(**kwargs)
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize_vector_store() first.")
        
        try:
            self.vector_store.delete(ids)
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def count(self) -> int:
        """Get total number of documents in the vector store"""
        
        if not self.vector_store:
            return 0
        
        try:
            return self.vector_store._collection.count()
        except:
            return 0
    
    def reset_collection(self) -> bool:
        """Reset/clear the entire collection"""
        
        if not self.vector_store:
            return False
        
        try:
            # Get collection name before deletion
            collection_name = self.vector_store._collection.name
            
            # Delete and recreate
            self.vector_store._client.delete_collection(collection_name)
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print(f"Collection '{collection_name}' reset successfully")
            return True
            
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        try:
            collection = self.vector_store._collection
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": getattr(collection, 'metadata', None)
            }
        except Exception as e:
            return {"error": str(e)}