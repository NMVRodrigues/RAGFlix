from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_core.messages import SystemMessage,HumanMessage
import json
import re

from langchain_ollama import ChatOllama

# Replace your current LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

"""
# Initialize the model
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False 
)

# Create LangChain components
llm_pipeline = HuggingFacePipeline(pipeline=pipe)
llm = ChatHuggingFace(llm=llm_pipeline)
"""

def custom_llm_with_tools(llm, tools, user_message):
    """Custom tool calling implementation for Llama 3.2."""
    
    # Format tools for Llama 3.2's expected JSON format
    tools_json = []
    for tool in tools:
        tool_dict = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for movies"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        tools_json.append(tool_dict)
    
    # Format the prompt according to your model's template
    formatted_prompt = f"""Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. Do not use variables.

{json.dumps(tools_json, indent=2)}

{user_message}"""
    
    # Call the LLM
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    
    # Parse the response to extract tool calls
    try:
        response_text = response.content.strip()
        
        # Look for JSON in the response
        json_match = re.search(r'\{[^}]*"name"[^}]*\}', response_text)
        if json_match:
            tool_call_json = json.loads(json_match.group())
            
            # Convert to LangChain format - handle both "retrieve" and "retrieve_movies"
            tool_name = tool_call_json["name"]
            #if tool_name in ["retrieve", "retrieve_movies"]:
            #    tool_name = "retrieve_movies"  # Normalize to correct name
            
            from langchain_core.messages import AIMessage
            tool_calls = [{
                "name": tool_name,
                "args": tool_call_json.get("parameters", {}),
                "id": f"call_{abs(hash(response_text)) % 10000}"
            }]
            
            return AIMessage(content="I'll search the movie database for you.", tool_calls=tool_calls)
    
    except Exception as e:
        print(f"Error parsing tool call: {e}")
        pass
    
    # If no tool call found, return regular response
    return response