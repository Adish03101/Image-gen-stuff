from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_community.llms import Ollama
from google import genai
import chromadb
from chromadb.config import Settings
import os

llm = Ollama(model="llama3.1", temperature=0.7, max_tokens=1024)

GEMNI_API_KEY =

class ImgGen(TypedDict):
    prompt: str
    base_prompt: List[str]
    doc_info: List[str]
    doc_embeddings: List[List[float]]
    prompt_embeddings: List[float]
    retrieve: List[str]
    n: int = 1
    size: str = "256x256"

def base_prompt(state: ImgGen) -> ImgGen:
    """
    Initializes the base prompt for the image generation.
    """
    state['base_prompt'] = [
        "You are an expert image generator. Your task is to create images based on the provided information.",
        "You will be given a set of documents, and you need to generate an image that best represents the content of these documents."
    ]
    return state

def embed(state: ImgGen) -> ImgGen:
    retrieval_folder = 'Docs'
    content = []
    for file in os.listdir(retrieval_folder):
        if file.endswith('.txt'):
            with open(os.path.join(retrieval_folder, file), 'r', encoding='utf-8') as f:
                content.append(f.read())
    client = genai.Client(api_key="GEMINI_API_KEY")
    result_doc = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=content,
        config={"task_type": "SEMANTIC_SIMILARITY"}
    )

    #because the base prompt is a list of strings,
    # we need to join them into a single string as it is small
    base_prompt_text = "\n".join(state['base_prompt'])
    result_prompt = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=[base_prompt_text],
        config={"task_type": "SEMANTIC_SIMILARITY"}
    )

    state['prompt_embeddings'] = result_prompt.embeddings[0]
    state['doc_embeddings'] = result_doc.embeddings
    state['doc_info'] = content  # Store the document texts here
    return state


def retrieve(state: ImgGen) -> ImgGen:
    """
    Retrieves the most relevant document using ChromaDB vector search.
    """
    # 1. Initialize ChromaDB client
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chromadb"  # Stores vectors locally
    ))
    
    # 2. Get or create collection
    collection = client.get_or_create_collection(name="image_gen_docs")

    # 3. Add documents and embeddings to ChromaDB (if not already done)
    # Note: ChromaDB expects IDs as strings and documents as a list of strings
    if collection.count() == 0:
        ids = [f"doc_{i}" for i in range(len(state['doc_info']))]
        collection.add(
            embeddings=state['doc_embeddings'],
            documents=state['doc_info'],
            ids=ids
        )

    # 4. Query ChromaDB with prompt embedding
    results = collection.query(
        query_embeddings=[state['prompt_embeddings']],
        n_results=1  # Get the most relevant document
    )

    # 5. Update state with the most relevant document
    if results and results['documents']:
        state['retrieval'] = results['documents'][0]
    else:
        state['retrieval'] = []

    return state

def llm_prompt(state: ImgGen) -> ImgGen:
    """
    Uses an LLM to generate a prompt for image generation.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    system = SystemMessage(
        content="You are an expert prompt engineer. Create a vivid, detailed prompt for an image generator based on the following information."
    )
    human = HumanMessage(content=f"Information:\n{state['retrieval'][0] if state['retrieval'] else 'No information available.'}")
    response = llm.invoke([system, human])
    state['prompt'] = response.content
    return state


def generate(state: ImgGen) -> ImgGen:
    """
    Generates the image using Hugging Face's Stable Diffusion.
    """
    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate images
    images = pipe(state['prompt'], num_images_per_prompt=state['n']).images

    # Convert first image to base64 string
    buffered = io.BytesIO()
    images[0].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Store the base64 string in state
    state['image'] = img_str
    return state

graph = StateGraph(ImgGen, start=START, end=END)
graph.add_node('embed', embed)
graph.add_node('retrieve', retrieve)  # <-- Correct spelling
graph.add_node('llm_prompt', llm_prompt)  # <-- Use LLM prompt node
graph.add_node('Image gen', generate)

# Add edges
graph.add_edge(START, 'embed')
graph.add_edge('embed', 'retrieve')
graph.add_edge('retrieve', 'llm_prompt')
graph.add_edge('llm_prompt', 'Image gen')
graph.add_edge('Image gen', END)

                
