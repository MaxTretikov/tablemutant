#!/usr/bin/env python3
"""
EmbeddingCache - Handles caching of document embeddings based on document hash
"""

import hashlib
import json
import os
import pickle
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class EmbeddingCache:
    def __init__(self):
        self.cache_dir = self._get_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_dir(self) -> Path:
        """Get the appropriate cache directory based on the platform."""
        system = platform.system()
        
        if system == "Linux":
            cache_dir = Path.home() / '.tablemutant' / 'embeddings'
        elif system == "Darwin":  # macOS
            cache_dir = Path.home() / 'Library' / 'Application Support' / 'TableMutant' / 'embeddings'
        elif system == "Windows":
            cache_dir = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local')) / 'TableMutant' / 'embeddings'
        else:
            # Fallback to home directory
            cache_dir = Path.home() / '.tablemutant' / 'embeddings'
        
        return cache_dir
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def _get_cache_file_path(self, file_hash: str) -> Path:
        """Get the cache file path for a given file hash."""
        return self.cache_dir / f"{file_hash}.pkl"
    
    def _get_metadata_file_path(self, file_hash: str) -> Path:
        """Get the metadata file path for a given file hash."""
        return self.cache_dir / f"{file_hash}_meta.json"
    
    def has_cached_embeddings(self, file_path: str) -> bool:
        """Check if embeddings are cached for the given file."""
        file_hash = self._calculate_file_hash(file_path)
        if not file_hash:
            return False
        
        cache_file = self._get_cache_file_path(file_hash)
        meta_file = self._get_metadata_file_path(file_hash)
        
        return cache_file.exists() and meta_file.exists()
    
    def get_cached_embeddings(self, file_path: str) -> Optional[Tuple[List[str], np.ndarray, Dict]]:
        """
        Retrieve cached embeddings for a file.
        
        Returns:
            Tuple of (text_chunks, embeddings, metadata) or None if not cached
        """
        file_hash = self._calculate_file_hash(file_path)
        if not file_hash:
            return None
        
        cache_file = self._get_cache_file_path(file_hash)
        meta_file = self._get_metadata_file_path(file_hash)
        
        if not (cache_file.exists() and meta_file.exists()):
            return None
        
        try:
            # Load embeddings
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Load metadata
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            text_chunks = cached_data.get('text_chunks', [])
            embeddings = cached_data.get('embeddings', np.array([]))
            
            print(f"Loaded cached embeddings for {os.path.basename(file_path)} ({len(text_chunks)} chunks)")
            return text_chunks, embeddings, metadata
            
        except Exception as e:
            print(f"Error loading cached embeddings for {file_path}: {e}")
            # Clean up corrupted cache files
            try:
                cache_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)
            except:
                pass
            return None
    
    def cache_embeddings(self, file_path: str, text_chunks: List[str], 
                        embeddings: np.ndarray, metadata: Dict = None) -> bool:
        """
        Cache embeddings for a file.
        
        Args:
            file_path: Path to the original file
            text_chunks: List of text chunks that were embedded
            embeddings: Numpy array of embeddings
            metadata: Optional metadata dictionary
        
        Returns:
            True if caching was successful, False otherwise
        """
        file_hash = self._calculate_file_hash(file_path)
        if not file_hash:
            return False
        
        cache_file = self._get_cache_file_path(file_hash)
        meta_file = self._get_metadata_file_path(file_hash)
        
        try:
            # Prepare cache data
            cache_data = {
                'text_chunks': text_chunks,
                'embeddings': embeddings
            }
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'file_path': file_path,
                'file_hash': file_hash,
                'file_name': os.path.basename(file_path),
                'num_chunks': len(text_chunks),
                'embedding_shape': embeddings.shape if hasattr(embeddings, 'shape') else None,
                'cached_at': str(Path(file_path).stat().st_mtime) if os.path.exists(file_path) else None
            })
            
            # Save embeddings
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Save metadata
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Cached embeddings for {os.path.basename(file_path)} ({len(text_chunks)} chunks)")
            return True
            
        except Exception as e:
            print(f"Error caching embeddings for {file_path}: {e}")
            # Clean up partial files
            try:
                cache_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)
            except:
                pass
            return False
    
    def clear_cache(self) -> bool:
        """Clear all cached embeddings."""
        try:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            for file in self.cache_dir.glob("*_meta.json"):
                file.unlink()
            print("Embedding cache cleared")
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict:
        """Get information about the current cache."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            meta_files = list(self.cache_dir.glob("*_meta.json"))
            
            total_size = sum(f.stat().st_size for f in cache_files + meta_files)
            
            return {
                'cache_dir': str(self.cache_dir),
                'num_cached_documents': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            print(f"Error getting cache info: {e}")
            return {
                'cache_dir': str(self.cache_dir),
                'num_cached_documents': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0
            }