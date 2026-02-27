from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient

# Load PDF
loader = PyPDFLoader("gpt-5-system-card.pdf")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}  # IMPORTANT for BGE

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Store in Qdrant
qdrant = Qdrant.from_documents(
    chunks,
    embeddings,
    url="http://localhost:6333",
    collection_name="gpt_db",
    prefer_grpc=False,
)

print("✅ Data successfully stored in Qdrant!")
