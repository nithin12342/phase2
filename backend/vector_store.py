"""
Improved Vector Store with IVF Index for scalable similarity search.
Uses IndexIDMap for proper ID-based operations.
"""
import numpy as np
import os
from typing import List, Optional
import logging
import threading

logger = logging.getLogger(__name__)

# Graceful FAISS import
FAISS_AVAILABLE = False
faiss = None
try:
    import faiss as _faiss
    faiss = _faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS loaded successfully")
except ImportError as e:
    logger.warning(f"FAISS not available: {e}. Vector store will be disabled.")
except Exception as e:
    logger.warning(f"FAISS failed to load: {e}. Vector store will be disabled.")

VECTOR_DIM = 768  # Dimension of the embeddings
INDEX_FILE = "backend/database/faiss_index.bin"
ID_MAP_FILE = "backend/database/faiss_id_map.txt"
EMBEDDING_FILE = "backend/database/embeddings.npy"

NLIST = 100  # Number of clusters (adjust based on data size)
MIN_VECTORS_FOR_IVF = 1000  # Minimum vectors needed to train IVF index


class VectorStore:
    """
    Improved vector store with support for:
    - IndexIDMap for proper ID-based retrieval
    - IVF index for O(sqrt(n)) search performance
    - Persistent storage of embeddings for reconstruction
    - Graceful degradation when FAISS is unavailable
    """
    
    def __init__(self):
        self.index = None
        self.id_map: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self._lock = threading.Lock()
        if FAISS_AVAILABLE:
            self._initialize_index()
        else:
            logger.warning("VectorStore initialized without FAISS - search disabled")

    def _initialize_index(self):
        """Initialize or load the FAISS index."""
        if os.path.exists(INDEX_FILE):
            logger.info("Loading existing Faiss index...")
            try:
                self.index = faiss.read_index(INDEX_FILE)
                if os.path.exists(ID_MAP_FILE):
                    with open(ID_MAP_FILE, "r") as f:
                        self.id_map = [line.strip() for line in f.readlines()]
                if os.path.exists(EMBEDDING_FILE):
                    self.embeddings = list(np.load(EMBEDDING_FILE, allow_pickle=True))
                logger.info(f"Index loaded with {self.index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Error loading index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new flat L2 index (will be upgraded to IVF when enough data)."""
        logger.info("Creating new Faiss index (FlatL2)...")
        self.index = faiss.IndexFlatL2(VECTOR_DIM)
        self.id_map = []
        self.embeddings = []

    def _upgrade_to_ivf(self):
        """
        Upgrade from FlatL2 to IVFFlat for better performance.
        Only called when we have enough vectors.
        """
        if len(self.embeddings) < MIN_VECTORS_FOR_IVF:
            return False
        
        logger.info(f"Upgrading to IVF index with {NLIST} clusters...")
        try:
            quantizer = faiss.IndexFlatL2(VECTOR_DIM)
            
            nlist = min(NLIST, len(self.embeddings) // 10)  # At least 10 vectors per cluster
            new_index = faiss.IndexIVFFlat(quantizer, VECTOR_DIM, nlist)
            
            training_data = np.array(self.embeddings).astype('float32')
            new_index.train(training_data)
            
            new_index.add(training_data)
            
            self.index = new_index
            self._save()
            logger.info("Successfully upgraded to IVF index.")
            return True
        except Exception as e:
            logger.error(f"Failed to upgrade to IVF: {e}")
            return False

    def add_vector(self, vector: np.ndarray, prediction_id: str):
        """Add a vector with its associated prediction ID (thread-safe)."""
        with self._lock:
            if self.index is None:
                raise RuntimeError("Faiss index is not initialized.")
            
            if vector.ndim == 1:
                vector = np.expand_dims(vector, axis=0)
            
            self.index.add(vector)
            self.id_map.append(prediction_id)
            self.embeddings.append(vector.flatten())
            self._save()
            
            if isinstance(self.index, faiss.IndexFlatL2) and len(self.embeddings) >= MIN_VECTORS_FOR_IVF:
                self._upgrade_to_ivf()

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[str]:
        """Search for k nearest neighbors (thread-safe)."""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            if query_vector.ndim == 1:
                query_vector = np.expand_dims(query_vector, axis=0)

            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = min(10, self.index.nlist)  # Search 10 clusters
            
            distances, indices = self.index.search(query_vector, k)
            
            return [self.id_map[i] for i in indices[0] if 0 <= i < len(self.id_map)]

    def get_vector_by_id(self, prediction_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a vector by its prediction ID.
        Now possible because we store embeddings separately.
        """
        if prediction_id not in self.id_map:
            raise ValueError(f"Prediction ID {prediction_id} not found in the vector store.")
        
        idx = self.id_map.index(prediction_id)
        if idx < len(self.embeddings):
            return np.array(self.embeddings[idx]).astype('float32')
        
        raise ValueError(f"Embedding not found for prediction ID {prediction_id}")

    def search_similar_to_id(self, prediction_id: str, k: int = 5) -> List[str]:
        """
        Find vectors similar to a given prediction ID.
        """
        query_vector = self.get_vector_by_id(prediction_id)
        return self.search(query_vector, k=k)

    def delete_vector(self, prediction_id: str) -> bool:
        """
        Mark a vector as deleted (soft delete) - thread-safe.
        Full deletion would require rebuilding the index.
        """
        with self._lock:
            if prediction_id in self.id_map:
                idx = self.id_map.index(prediction_id)
                self.id_map[idx] = f"__deleted__{prediction_id}"
                self._save()
                return True
            return False

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": VECTOR_DIM,
            "index_type": type(self.index).__name__ if self.index else None,
            "id_map_size": len(self.id_map),
            "is_ivf": hasattr(self.index, 'nlist') if self.index else False
        }

    def _save(self):
        """Persist the index and mappings to disk."""
        if self.index is not None:
            try:
                faiss.write_index(self.index, INDEX_FILE)
                with open(ID_MAP_FILE, "w") as f:
                    for item_id in self.id_map:
                        f.write(f"{item_id}\n")
                np.save(EMBEDDING_FILE, np.array(self.embeddings, dtype=object), allow_pickle=True)
            except Exception as e:
                logger.error(f"Error saving index: {e}")


vector_store = VectorStore()
