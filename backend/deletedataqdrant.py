from qdrant_client import QdrantClient
from qdrant_client.models import Filter

# Connect to your local Qdrant instance
client = QdrantClient(url="http://localhost:6333")

collection_name = "knowledge_base"

# Delete all points (empty filter = delete all)
client.delete(
    collection_name=collection_name,
    points_selector=Filter(must=[])  # Correct way to specify "delete all"
)

print(f"✅ All data deleted from collection '{collection_name}', but the collection itself is preserved.")
        