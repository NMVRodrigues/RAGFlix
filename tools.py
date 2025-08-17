
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.tools import tool

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from langchain_core.documents import Document
from typing_extensions import List

from langchain_ollama import ChatOllama

# Replace your current LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

class State(MessagesState):
    context: List[Document]

def query_or_respond(state: State):
    """Generate tool call for movie retrieval or respond directly."""
    
    # Add system message to encourage tool usage for movie queries
    system_message = SystemMessage(content=(
        "You are a movie recommendation assistant. When users ask about movies, "
        "you should use the retrieve tool to search the movie database "
        "before providing recommendations. Only recommend movies found in the database."
    ))
    
    # Combine system message with conversation history
    messages_with_system = [system_message] + state["messages"]
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(messages_with_system)
    
    return {"messages": [response]}

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    # Increase k to get more movies when user asks for multiple recommendations
    retrieved_docs = vector_store.similarity_search(query, k=6)  # Increased from 2 to 6
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    """Generate movie recommendation response."""
    
    # Get the most recent tool messages (movie retrieval results)
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]  # Reverse to get correct order
    
    # Format movie information for the context
    movies_content = "\n\n".join(msg.content for msg in tool_messages)
    
    # Get the most recent user question
    user_question = ""
    for message in reversed(state["messages"]):
        if message.type == "human":
            user_question = message.content
            break
    
    # Simplified generation prompt
    generation_prompt = f"""You are a movie recommendation assistant. Based on the user's query and the provided movie database context, provide detailed movie recommendations.

User Query: {user_question}

Available Movies from Database:
{movies_content}

Instructions:
- Recommend movies that best match the user's preferences
- If they ask for multiple movies, recommend multiple movies from the database
- Include relevant details like title, year, director, cast, genres, and plot
- Explain why each movie matches their request
- Only recommend movies that appear in the database context above
- Format your response clearly and engagingly

Provide your recommendations now:"""

    # Generate response using the LLM directly
    response = llm.invoke([HumanMessage(content=generation_prompt)])
    
    # Extract context from tool message artifacts for state tracking
    context = []
    for tool_message in tool_messages:
        if hasattr(tool_message, 'artifact') and tool_message.artifact:
            context.extend(tool_message.artifact)
    
    return {
        "messages": [response], 
        "context": context
    }