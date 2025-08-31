#!/usr/bin/env python3
"""
RAGProcessor - Handles RAG functionality for PDF loading, text extraction, and embedding generation
"""

import os
import re
import logging
from typing import List, Optional, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embedding_cache import EmbeddingCache

# Get logger for this module
logger = logging.getLogger('tablemutant.core.rag_processor')


class RAGProcessor:
    def __init__(self):
        self.embedding_cache = EmbeddingCache()
        self._sentence_transformers = None
        self._embedding_model = None
    
    def load_rag_source(self, file_path: str) -> Optional[str]:
        """Load and extract text from various document formats for RAG."""
        try:
            # Determine file type based on extension
            file_ext = file_path.lower().split('.')[-1]
            
            if file_ext == 'pdf':
                return self._load_pdf(file_path)
            elif file_ext in ['txt', 'md']:
                return self._load_text_file(file_path)
            else:
                logger.warning("Unsupported file type: %s", file_ext)
                return None
                
        except Exception as e:
            logger.error("Error reading file %s: %s", file_path, e)
            return None
    
    def _load_pdf(self, pdf_path: str) -> Optional[str]:
        """Load and extract text from PDF."""
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return None
            
    def _load_text_file(self, file_path: str) -> Optional[str]:
        """Load text from plain text or markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception:
                return None
    
    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                # Try to import sentence-transformers
                if self._sentence_transformers is None:
                    import sentence_transformers
                    self._sentence_transformers = sentence_transformers
                
                # Use a lightweight, fast model for embeddings
                model_name = "all-MiniLM-L6-v2"  # Small, fast, good quality
                logger.info("Loading embedding model: %s", model_name)
                self._embedding_model = self._sentence_transformers.SentenceTransformer(model_name)
                logger.info("Embedding model loaded successfully")
                
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                return None
            except Exception as e:
                logger.error("Error loading embedding model: %s", e)
                return None
        
        return self._embedding_model
    
    def _chunk_text(self, text: str, chunk_type: str = "sentence") -> List[str]:
        """Split text into chunks for embedding - each chunk should be a sentence or line."""
        if not text or not text.strip():
            return []
        
        chunks = []
        
        if chunk_type == "sentence":
            # Split by sentences using multiple delimiters
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Filter out very short fragments
                    chunks.append(sentence)
        
        elif chunk_type == "line":
            # Split by lines (newlines)
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # Filter out very short lines
                    chunks.append(line)
        
        else:  # "paragraph" or fallback
            # Split by double newlines (paragraphs) then by sentences
            paragraphs = re.split(r'\n\s*\n', text)
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Split paragraph into sentences
                sentences = re.split(r'[.!?]+', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10:
                        chunks.append(sentence)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def generate_embeddings(self, file_path: str, force_regenerate: bool = False, progress_callback=None) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Generate embeddings for a document, using cache if available.
        
        Args:
            file_path: Path to the document
            force_regenerate: If True, regenerate even if cached
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Tuple of (text_chunks, embeddings) or None if failed
        """
        # Check cache first unless forced to regenerate
        if not force_regenerate and self.embedding_cache.has_cached_embeddings(file_path):
            cached_result = self.embedding_cache.get_cached_embeddings(file_path)
            if cached_result:
                text_chunks, embeddings, metadata = cached_result
                return text_chunks, embeddings
        
        # Load and extract text
        if progress_callback:
            progress_callback(10, "Loading document...", None, None)
        
        text = self.load_rag_source(file_path)
        if not text:
            logger.error("Failed to extract text from %s", file_path)
            return None
        
        if progress_callback:
            progress_callback(20, "Loading embedding model...", None, None)
        
        # Get embedding model
        model = self._get_embedding_model()
        if model is None:
            logger.error("Embedding model not available")
            return None
        
        try:
            # Chunk the text
            if progress_callback:
                progress_callback(30, "Chunking text...", None, None)
            
            logger.info("Chunking text from %s...", os.path.basename(file_path))
            text_chunks = self._chunk_text(text)
            
            if not text_chunks:
                logger.warning("No text chunks generated")
                return None
            
            logger.info("Generated %s text chunks", len(text_chunks))
            
            if progress_callback:
                progress_callback(40, f"Generating embeddings for {len(text_chunks)} chunks...", None, None)
            
            # Generate embeddings with custom progress tracking
            logger.info("Generating embeddings...")
            
            # Create a custom progress callback for sentence-transformers
            def embedding_progress_callback(current_batch, total_batches):
                if progress_callback:
                    # Map embedding progress to 40-90% range
                    embedding_progress = (current_batch / total_batches) * 50 + 40
                    progress_callback(embedding_progress, f"Processing batch {current_batch}/{total_batches}...", current_batch, total_batches)
            
            # Use custom encoding with progress tracking
            embeddings = self._encode_with_progress(model, text_chunks, embedding_progress_callback)
            
            # Convert to numpy array if not already
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            logger.info("Generated embeddings with shape: %s", embeddings.shape)
            
            if progress_callback:
                progress_callback(95, "Caching embeddings...", None, None)
            
            # Cache the results
            metadata = {
                'model_name': model.get_sentence_embedding_dimension() if hasattr(model, 'get_sentence_embedding_dimension') else 'unknown',
                'chunk_type': 'sentence',
                'min_chunk_length': 10
            }
            
            success = self.embedding_cache.cache_embeddings(file_path, text_chunks, embeddings, metadata)
            if success:
                logger.info("Embeddings cached for %s", os.path.basename(file_path))
            else:
                logger.warning("Failed to cache embeddings for %s", os.path.basename(file_path))
            
            if progress_callback:
                progress_callback(100, "Complete!", None, None)
            
            return text_chunks, embeddings
            
        except Exception as e:
            logger.error("Error generating embeddings for %s: %s", file_path, e)
            return None
    
    def _encode_with_progress(self, model, text_chunks, progress_callback=None):
        """Encode text chunks with progress tracking."""
        try:
            # Try to use batch processing with progress tracking
            batch_size = getattr(model, 'batch_size', 32)  # Default batch size
            total_batches = (len(text_chunks) + batch_size - 1) // batch_size
            
            all_embeddings = []
            
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i + batch_size]
                current_batch = (i // batch_size) + 1
                
                if progress_callback:
                    progress_callback(current_batch, total_batches)
                
                # Encode this batch
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
            
            return np.array(all_embeddings)
            
        except Exception as e:
            logger.error("Error in batch encoding: %s", e)
            # Fallback to original method
            return model.encode(text_chunks, show_progress_bar=True)
    
    def get_cached_embeddings(self, file_path: str) -> Optional[Tuple[List[str], np.ndarray, Dict]]:
        """Get cached embeddings for a file."""
        return self.embedding_cache.get_cached_embeddings(file_path)
    
    def has_cached_embeddings(self, file_path: str) -> bool:
        """Check if embeddings are cached for a file."""
        return self.embedding_cache.has_cached_embeddings(file_path)
    
    def clear_embedding_cache(self) -> bool:
        """Clear all cached embeddings."""
        return self.embedding_cache.clear_cache()
    
    def get_cache_info(self) -> Dict:
        """Get information about the embedding cache."""
        return self.embedding_cache.get_cache_info()
    
    def find_relevant_chunks(self, query_text: str, rag_embeddings_data: List[Dict], top_k: int = 25) -> List[str]:
        """
        Find the most relevant text chunks using semantic search.
        
        Args:
            query_text: The text to search for (e.g., row data)
            rag_embeddings_data: List of embedding data from documents
            top_k: Number of top chunks to return
            
        Returns:
            List of most relevant text chunks
        """
        if not rag_embeddings_data or not query_text.strip():
            return []
        
        try:
            # Get embedding model
            model = self._get_embedding_model()
            if model is None:
                logger.error("Embedding model not available for semantic search")
                return []
            
            # Generate embedding for the query
            query_embedding = model.encode([query_text])
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            # Collect all chunks and their embeddings
            all_chunks = []
            all_embeddings = []
            
            for doc_data in rag_embeddings_data:
                chunks = doc_data.get('chunks', [])
                embeddings = doc_data.get('embeddings', np.array([]))
                doc_path = doc_data.get('path', 'unknown')
                
                if len(chunks) > 0 and embeddings.size > 0:
                    # Add document context to chunks
                    doc_name = os.path.basename(doc_path)
                    contextualized_chunks = [f"[{doc_name}] {chunk}" for chunk in chunks]
                    
                    all_chunks.extend(contextualized_chunks)
                    all_embeddings.extend(embeddings)
            
            if not all_chunks or not all_embeddings:
                return []
            
            # Convert to numpy array
            all_embeddings = np.array(all_embeddings)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, all_embeddings)[0]
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return the most relevant chunks
            relevant_chunks = [all_chunks[i] for i in top_indices]
            
            logger.info("Found %s relevant chunks from %s documents", len(relevant_chunks), len(rag_embeddings_data))
            top_scores = [similarities[i] for i in top_indices[:5]]
            logger.debug("Top similarity scores: %s", [f'{score:.3f}' for score in top_scores])
            
            return relevant_chunks
            
        except Exception as e:
            logger.error("Error in semantic search: %s", e)
            # Fallback: return first few chunks from each document
            fallback_chunks = []
            for doc_data in rag_embeddings_data:
                chunks = doc_data.get('chunks', [])
                doc_path = doc_data.get('path', 'unknown')
                doc_name = os.path.basename(doc_path)
                
                # Take first few chunks from each document
                for chunk in chunks[:5]:  # Limit per document
                    fallback_chunks.append(f"[{doc_name}] {chunk}")
                    if len(fallback_chunks) >= top_k:
                        break
                if len(fallback_chunks) >= top_k:
                    break
            
            return fallback_chunks[:top_k]