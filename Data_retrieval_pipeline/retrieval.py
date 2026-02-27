from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

# Embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Connect to existing collection
db = Qdrant(client=client, embeddings=embeddings, collection_name="gpt_db")

query = "what is gpt-5?"

# Perform similarity search
docs = db.similarity_search(query, k=3)

# Print results
for i, doc in enumerate(docs):
    print(f"\nResult {i + 1}:")
    print(doc.page_content[:500])
