from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama

# Embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Connect to collection
db = Qdrant(client=client, embeddings=embeddings, collection_name="gpt_db")

query = "what is gpt-5?"

# Retrieve
docs = db.similarity_search(query, k=3)

retrieved_docs = ""
for i, doc in enumerate(docs):
    print(f"\nResult {i + 1}:")
    print(doc.page_content[:500])
    retrieved_docs += doc.page_content

# Build prompt
combined_input = f"""
Based on the following documents, answer the question.

Question: {query}

Documents:
{retrieved_docs}

If the answer is not in the documents, say you don't have enough information.
"""

#  Ollama model
model = ChatOllama(model="llama3", base_url="http://localhost:11434", temperature=0)

# Invoke directly (no messages list needed)
result = model.invoke(combined_input)

print("\nAnswer:")
print(result.content)
