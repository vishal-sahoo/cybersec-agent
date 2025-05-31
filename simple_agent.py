from dotenv import load_dotenv  # Import load_dotenv

# Load environment variables from .env file
# IMPORTANT: This should be one of the first lines of your script
load_dotenv()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import (
    GoogleGenAI,
)
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- Configuration ---
GEMINI_MODEL_NAME = "gemini-pro"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "./chroma_db_store"
CHROMA_COLLECTION_NAME = "my_rag_collection"
DATA_DIR = "./data"

# --- 1. Setup LLM and Embedding Model (using LlamaIndex Settings) ---
print(f"Setting up LLM: Google Generative AI ({GEMINI_MODEL_NAME})")

# The GoogleGenAI class will automatically look for the GOOGLE_API_KEY
# in the environment variables, which load_dotenv() has now populated.
# You could also explicitly pass it like this, though it's often not needed:
# api_key_from_env = os.getenv("GOOGLE_API_KEY")
# if not api_key_from_env:
#     print("Error: GOOGLE_API_KEY not found in .env file or environment.")
#     exit()
# Settings.llm = GoogleGenAI(model_name=GEMINI_MODEL_NAME, api_key=api_key_from_env)

Settings.llm = GoogleGenAI(
    model_name=GEMINI_MODEL_NAME
    # GOOGLE_API_KEY environment variable (now loaded from .env) will be used by default
)

print(f"Setting up Embedding Model: HuggingFace ({EMBEDDING_MODEL_NAME})")
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME, device="cpu"
)

# --- 2. Load Documents ---
print(f"Loading documents from: {DATA_DIR}")
try:
    import pathlib

    pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Please ensure you have .txt files in the '{DATA_DIR}' directory.")

    reader = SimpleDirectoryReader(DATA_DIR, required_exts=[".txt"])
    documents = reader.load_data()
    if not documents:
        print(
            f"No .txt documents found in '{DATA_DIR}'. The agent will have no knowledge."
        )
    else:
        print(f"Loaded {len(documents)} document(s).")
except Exception as e:
    print(f"Error loading documents: {e}")
    exit()

# --- 3. Setup ChromaDB Vector Store ---
print(f"Setting up ChromaDB vector store at: {CHROMA_DB_PATH}")
db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# --- 4. Create the Index ---
print("Creating or updating index (this may take a moment for new documents)...")
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
)
print("Index created/updated successfully.")

# --- 5. Create a Query Engine ---
print("Creating query engine...")
query_engine = index.as_query_engine(similarity_top_k=3)
print("Query engine created.")

# --- 6. Query the Agent ---
print("\n--- RAG Agent Ready ---")
print("Ask questions about your documents. Type 'exit' to quit.")

while True:
    user_query = input("Your query: ")
    if user_query.lower() == "exit":
        break
    if not user_query.strip():
        continue

    print("Processing your query...")
    try:
        response = query_engine.query(user_query)
        print("\nAgent Response:")
        print(str(response))
        print("\n--------------------\n")
    except Exception as e:
        print(f"Error during query processing: {e}")
        # You might want to print the full traceback for debugging
        # import traceback
        # traceback.print_exc()

print("Exiting agent.")
