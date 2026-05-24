import os
import hashlib
from typing import List, Dict, Any, Optional

class VectorStore:
    """In-memory vector store for development. Replace with Pinecone in production."""
    
    def __init__(self):
        # For development: store vectors in memory
        # In production, use Pinecone
        self.memories = []
        self.use_pinecone = False
        
        # Try to initialize Pinecone
        try:
            import pinecone
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if pinecone_api_key:
                pinecone.init(api_key=pinecone_api_key, environment="us-west1-aws")
                self.index_name = os.getenv("PINECONE_INDEX_NAME", "continuum-memory")
                
                # Create index if it doesn't exist
                if self.index_name not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=self.index_name,
                        dimension=1536,  # OpenAI embedding dimension
                        metric="cosine"
                    )
                
                self.index = pinecone.Index(self.index_name)
                self.use_pinecone = True
                print(f"Pinecone initialized with index: {self.index_name}")
        except Exception as e:
            print(f"Pinecone not available, using in-memory store: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text. In production, use OpenAI's embedding API."""
        # Simple mock embedding for development
        # Replace with: openai.Embedding.create(input=text, model="text-embedding-3-small")
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()[:128]
        return [float(b) / 255.0 for b in hash_bytes]
    
    def add_memory(self, session_id: str, text: str, metadata: Dict[str, Any]):
        """Store a memory in the vector store."""
        memory_id = hashlib.md5(f"{session_id}{text}{len(self.memories)}".encode()).hexdigest()
        
        if self.use_pinecone:
            # Store in Pinecone
            embedding = self._get_embedding(text)
            self.index.upsert(vectors=[(
                memory_id,
                embedding,
                {**metadata, "session_id": session_id, "text": text[:500]}
            )])
        else:
            # Store in memory (development)
            self.memories.append({
                "id": memory_id,
                "session_id": session_id,
                "text": text,
                "metadata": metadata,
                "embedding": self._get_embedding(text)
            })
    
    def search(self, query: str, session_id: str, limit: int = 10) -> List[Dict]:
        """Search for similar memories."""
        query_embedding = self._get_embedding(query)
        results = []
        
        if self.use_pinecone:
            # Search in Pinecone
            response = self.index.query(
                vector=query_embedding,
                filter={"session_id": session_id},
                top_k=limit,
                include_metadata=True
            )
            for match in response.matches:
                results.append({
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "metadata": match.metadata
                })
        else:
            # Search in memory (development)
            for mem in self.memories:
                if mem["session_id"] != session_id:
                    continue
                # Simple cosine similarity
                similarity = self._cosine_similarity(query_embedding, mem["embedding"])
                results.append({
                    "score": similarity,
                    "text": mem["text"],
                    "metadata": mem["metadata"]
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:limit]
        
        return results
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0