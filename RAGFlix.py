import os
import pandas as pd
from VectorDB import VectorDB
from utils.GenerateDocuments import create_documents_from_dataset

def main():
    # Configuration
    PERSIST_DIR = "./chroma_langchain_db"
    COLLECTION_NAME = "movies_collection"
    MIN_EXPECTED_DOCS = 9000     # Minimum documents to consider collection "ready"
    
    # Load dataset
    df = pd.read_csv(os.path.join('movie_data', 'movies_2020-01-01_2025-01-01.csv'))
    print(f"Dataset loaded: {len(df)} movies")

    # Initialize vector database
    vector_db = VectorDB(model_name="BAAI/bge-base-en-v1.5", batch_size=32)
    init_result = vector_db.initialize_vector_store(PERSIST_DIR, COLLECTION_NAME)

    
    # Check existing documents
    existing_count = vector_db.count()
    
    if existing_count >= MIN_EXPECTED_DOCS:
        print(f"Found existing collection with {existing_count} documents - using existing data")
    else:
        if existing_count > 0:
            print(f"Found partial collection with {existing_count} documents - resetting and rebuilding")
            vector_db.reset_collection()
        else:
            print("No existing collection found - creating new one")
        
        # Generate and add documents
        print("Generating documents...")
        documents = create_documents_from_dataset(df)
        print(f"Generated {len(documents)} documents")
        
        # Use subset for testing
        print(f"Adding {len(documents)} documents to collection")
        
        final_count = vector_db.add_documents(documents, batch_size=256)
        print(f"Documents added successfully. Total: {final_count}")

    # Continue with tests...
    run_tests(vector_db)

def run_tests(vector_db: VectorDB):
    """Run similarity search tests"""
    
    # Display collection info
    collection_info = vector_db.get_collection_info()
    print(f"\nCollection info: {collection_info}")

    # Test similarity search
    print("\n=== Testing Similarity Search ===")
    test_queries = [
        "movies about artificial intelligence",
        "romantic comedies",
        "action movies with explosions",
        "space exploration films"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = vector_db.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            title = doc.metadata.get('title', 'Unknown')
            year = doc.metadata.get('year', 'Unknown')
            genre = doc.metadata.get('genre', 'Unknown')
            print(f"  {i}. {title} ({year}) - {genre}")

    # Test filtered search
    print("\n=== Testing Filtered Search ===")
    results = vector_db.similarity_search(
        query="space adventure",
        k=5,
        filter={"genre": "sci-fi"}
    )
    print(f"Sci-fi space adventures ({len(results)} found):")
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get('title', 'Unknown')
        year = doc.metadata.get('year', 'Unknown')
        print(f"  {i}. {title} ({year})")

    print("\n=== All tests completed successfully! ===")

if __name__ == "__main__":
    main()