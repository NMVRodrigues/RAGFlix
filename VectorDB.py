from langchain_elasticsearch import ElasticsearchVectorStore
import os
from typing import Optional



class VectorDB():
    """
    A class that handles interactions with a vector database.
    Only Elasticsearch is supported at the moment.
    Future plans include chromadb and other vector databases.
    """
    def __init__(self, *args, **kwargs):
        self.vector_store = None
    
    def check_vector_store(self, vectordb_path: str):
        """
        Checks if the vector store exists and if it does how many documents it contains.
        """

        # check if path exists
        if not os.path.exists(vectordb_path):
            raise ValueError(f"Vector store path '{vectordb_path}' does not exist.")
        # check if vector store is initialized
        else:
            # check how many documents are in the vector store
            num_docs = self.vector_store.count()
            return f"Vector store contains {num_docs} documents."
            
    def initialize_vector_store(self, vectordb_path: str, index_name: Optional[str] = None):
        """
        Initializes the vector store with the given path and index name.
        If index_name is not provided, it defaults to 'documents'.
        """
        if not self.check_vector_store(vectordb_path):
            return  # Stop execution if vector store is not valid

        self.vector_store = ElasticsearchVectorStore(
            url=vectordb_path,
            index_name=index_name or 'documents'
        )
        return f"Vector store initialized at {vectordb_path} with index '{self.vector_store.index_name}'."
    
    
