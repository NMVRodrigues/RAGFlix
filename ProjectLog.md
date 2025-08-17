# Log

#### Starting Point

Gathering data for the project.  
Movie and TV show data from IMDB and TMDB. Missing movie poster data from TMBD, will be added later.  
Next steps will be to check the data, clean if necessary, look at token/character limit to ensure it fits within the embedding model's constraints, and start developing the vector DB.  

#### VectorDb created

Created a vectorDb (chroma) that uses BGE embeddings (small enough to run on 11Gb VRAM household gpu) and tested the embedding and retrieval.  
Still need to check complex queries using tag filters.  
Might add an option to toggle the embedding model for pre M1 mac users that dont have dedicated GPU and need a faster solution.

Next steps will be to add the complex queries and get working on some kind of UI for the interaction with the system. If it can be used than what is the point?


#### Generation and Graph improvement

Changed the graph structure so that it can now be almost a conversational agent, capable of handling more complex interactions and maintaining context over multiple queries. It now can make the choice between retrieving information, whenever necessary, or generating responses based on the context it has.

Had to change the generation llm, from a huggingface llama model to a ollama llama model, simply because they llms loaded through hugginface where not being able to execute tool calls from langchain.

So, right now what we have a functioning conversational RAG system.

TODO list:
- Add a UI for the interaction with the system, so that it can be used by anyone.
- Implement complex query handling with tag filters (self-querying or something along those lines)
- Maybe a reranker, altough not sure if needed, but still
- Transform this into an agentic system, with an orchestrator agent, which routes the query for a movie or a series agent, based on the query.
