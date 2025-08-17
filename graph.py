from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from tools import query_or_respond, generate, retrieve, State

def create_movie_rag_graph():
    """Create and compile the movie RAG graph."""
    
    # Create tools node
    tools = ToolNode([retrieve])
    
    # Build the graph
    graph_builder = StateGraph(State)
        
    # Add nodes
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", tools)
    graph_builder.add_node("generate", generate)
    
    # Set entry point
    graph_builder.set_entry_point("query_or_respond")
    
    # Add conditional edges
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    
    # Add regular edges
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)
    
    # Compile and return the graph
    return graph_builder.compile()

# Create the graph instance
movie_rag_graph = create_movie_rag_graph()