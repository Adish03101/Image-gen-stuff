from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_community.llms import Ollama
import os

llm = Ollama(model="llama3.1", temperature=0.7, max_tokens=1024)

class ImgGen(TypedDict):
    prompt: str
    retrieval: List[str]
    n: int = 1
    size: str = "256x256"

def rag(state: ImgGen) -> ImgGen:
    """
    Retrieves relevant information for the image generation prompt.
    """
    retreival_folder = 'Docs'
    content = []
    
    for file in os.listdir(retreival_folder):
        if file.endswith('.txt'):
            with open(os.path.join(retreival_folder, file), 'r') as f:
                content.append(f.read())
    client = genai.Client(api_key="GEMINI_API_KEY")

# Embed your prompt or document chunks
    result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=["Your prompt or document text"],
        config={"task_type": "SEMANTIC_SIMILARITY"}
    )
    embeddings = result.embeddings

def prompt(state: ImgGen) -> ImgGen:
    """
    Constructs the prompt for the image generation from the retrieved document.
    """
    state['prompt'] = f"Generate an image based on the following information: {state['retrieval']}"
    return state

def generate(state: ImgGen) -> ImgGen:
    """
    Generates the image using the LLM.
    """
    response = llm.invoke(HumanMessage(content=state['prompt']))
    state['image'] = response.content
    return state

graph = StateGraph(ImgGen, start=START, end=END)
graph.add_node('rag',rag)
graph.add_node('prompt', prompt)
graph.add_node('Image gen', generate)

graph.add_edge(START, 'rag')
graph.add_edge('rag', 'prompt')
graph.add_edge('prompt', 'Image gen')
graph.add_edge('Image gen', END)


                
