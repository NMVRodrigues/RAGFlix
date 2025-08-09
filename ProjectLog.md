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