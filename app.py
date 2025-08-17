import chainlit as cl
from graph import movie_rag_graph
from VectorDB import VectorDB

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    # Ensure database is initialized
    PERSIST_DIR = "./chroma_langchain_db"
    COLLECTION_NAME = "movies_collection"

    # Initialize vector database
    vector_db = VectorDB(model_name="BAAI/bge-base-en-v1.5", batch_size=32)
    init_result = vector_db.initialize_vector_store(PERSIST_DIR, COLLECTION_NAME)

    vector_store = vector_db.vector_store
    
    await cl.Message(
        content="🎬 Welcome to MovieFlix RAG! I can help you discover amazing movies from our database. Try asking me for recommendations like:\n\n• 'Recommend some 1980s sci-fi movies'\n• 'Find me good action movies from the 90s'\n• 'What are some classic thrillers?'"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    
    # Show thinking message
    thinking_msg = cl.Message(content="🎬 Searching for movies...")
    await thinking_msg.send()
    
    try:
        # Create the state for your graph
        initial_state = {
            "messages": [{"role": "user", "content": message.content}]
        }
        
        # Run the graph
        result = movie_rag_graph.invoke(initial_state)
        
        # Get the final response
        final_message = result["messages"][-1]
        
        # Send the recommendation
        await cl.Message(content=final_message.content).send()
        
        # Show additional info about retrieved movies
        retrieved_count = len(result.get("context", []))
        if retrieved_count > 0:
            await cl.Message(
                content=f"💡 Based on {retrieved_count} movies from our database",
                author="System"
            ).send()
            
    except Exception as e:
        await cl.Message(
            content=f"❌ Sorry, I encountered an error: {str(e)}\n\nPlease try rephrasing your request."
        ).send()

if __name__ == "__main__":
    cl.run()