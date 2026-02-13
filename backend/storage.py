"""
Storage abstraction module for the Multimodal Deep Learning API.
Supports local filesystem and Azure Blob Storage.
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, BinaryIO
from pathlib import Path

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save(self, file_path: str, content: bytes) -> str:
        """Save content to storage. Returns the storage path/URL."""
        pass
    
    @abstractmethod
    def load(self, file_path: str) -> bytes:
        """Load content from storage."""
        pass
    
    @abstractmethod
    def delete(self, file_path: str) -> bool:
        """Delete file from storage."""
        pass
    
    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    def get_url(self, file_path: str) -> str:
        """Get URL/path for accessing the file."""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str = "backend/uploads"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _full_path(self, file_path: str) -> Path:
        """Get full path for a file."""
        return self.base_path / file_path
    
    def save(self, file_path: str, content: bytes) -> str:
        """Save content to local filesystem."""
        full_path = self._full_path(file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, "wb") as f:
            f.write(content)
        
        logger.debug(f"Saved file to {full_path}")
        return str(full_path)
    
    def load(self, file_path: str) -> bytes:
        """Load content from local filesystem."""
        full_path = self._full_path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(full_path, "rb") as f:
            return f.read()
    
    def delete(self, file_path: str) -> bool:
        """Delete file from local filesystem."""
        full_path = self._full_path(file_path)
        
        try:
            if full_path.exists():
                full_path.unlink()
                logger.debug(f"Deleted file: {full_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete {full_path}: {e}")
            return False
    
    def exists(self, file_path: str) -> bool:
        """Check if file exists on local filesystem."""
        return self._full_path(file_path).exists()
    
    def get_url(self, file_path: str) -> str:
        """Get local path for the file."""
        return str(self._full_path(file_path))


class AzureBlobStorage(StorageBackend):
    """Azure Blob Storage backend."""
    
    def __init__(self, connection_string: str, container_name: str):
        self.container_name = container_name
        self._connection_string = connection_string
        self._client = None
        self._container_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            self._client = BlobServiceClient.from_connection_string(self._connection_string)
            self._container_client = self._client.get_container_client(self.container_name)
            
            if not self._container_client.exists():
                self._container_client.create_container()
            
            logger.info(f"Connected to Azure Blob container: {self.container_name}")
        except ImportError:
            logger.warning("azure-storage-blob not installed. Install with: pip install azure-storage-blob")
            self._client = None
        except Exception as e:
            logger.warning(f"Failed to connect to Azure Blob Storage: {e}")
            self._client = None
    
    @property
    def is_connected(self) -> bool:
        """Check if Azure client is available."""
        return self._container_client is not None
    
    def save(self, file_path: str, content: bytes) -> str:
        """Save content to Azure Blob Storage."""
        if not self.is_connected:
            raise RuntimeError("Azure Blob client not connected")
        
        blob_client = self._container_client.get_blob_client(file_path)
        blob_client.upload_blob(content, overwrite=True)
        logger.debug(f"Saved file to Azure Blob: {file_path}")
        return blob_client.url
    
    def load(self, file_path: str) -> bytes:
        """Load content from Azure Blob Storage."""
        if not self.is_connected:
            raise RuntimeError("Azure Blob client not connected")
        
        blob_client = self._container_client.get_blob_client(file_path)
        return blob_client.download_blob().readall()
    
    def delete(self, file_path: str) -> bool:
        """Delete file from Azure Blob Storage."""
        if not self.is_connected:
            return False
        
        try:
            blob_client = self._container_client.get_blob_client(file_path)
            blob_client.delete_blob()
            logger.debug(f"Deleted file from Azure Blob: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Azure Blob: {e}")
            return False
    
    def exists(self, file_path: str) -> bool:
        """Check if file exists in Azure Blob Storage."""
        if not self.is_connected:
            return False
        
        blob_client = self._container_client.get_blob_client(file_path)
        return blob_client.exists()
    
    def get_url(self, file_path: str) -> str:
        """Get URL for Azure Blob file."""
        if not self.is_connected:
            return ""
        
        blob_client = self._container_client.get_blob_client(file_path)
        return blob_client.url


class StorageManager:
    """
    Storage manager that selects the appropriate backend based on configuration.
    Supports local filesystem and Azure Blob Storage.
    """
    
    def __init__(self):
        self._backend = self._initialize_backend()
    
    def _initialize_backend(self) -> StorageBackend:
        """Initialize the storage backend based on settings."""
        storage_type = getattr(settings, 'storage_backend', 'local')
        
        if storage_type == "azure":
            connection_string = getattr(settings, 'azure_storage_connection_string', None)
            container = getattr(settings, 'azure_container_name', None)
            
            if connection_string and container:
                try:
                    return AzureBlobStorage(connection_string, container)
                except Exception as e:
                    logger.warning(f"Failed to initialize Azure Blob, falling back to local: {e}")
        
        return LocalStorage()
    
    @property
    def backend(self) -> StorageBackend:
        """Get the active storage backend."""
        return self._backend
    
    def save(self, file_path: str, content: bytes) -> str:
        """Save content using the active backend."""
        return self._backend.save(file_path, content)
    
    def load(self, file_path: str) -> bytes:
        """Load content using the active backend."""
        return self._backend.load(file_path)
    
    def delete(self, file_path: str) -> bool:
        """Delete file using the active backend."""
        return self._backend.delete(file_path)
    
    def exists(self, file_path: str) -> bool:
        """Check if file exists using the active backend."""
        return self._backend.exists(file_path)
    
    def get_url(self, file_path: str) -> str:
        """Get URL for file using the active backend."""
        return self._backend.get_url(file_path)
    
    def get_backend_info(self) -> dict:
        """Get information about the active backend."""
        return {
            "type": type(self._backend).__name__,
            "base_path": getattr(self._backend, 'base_path', None),
            "bucket": getattr(self._backend, 'bucket_name', None),
            "container": getattr(self._backend, 'container_name', None)
        }


storage = StorageManager()
