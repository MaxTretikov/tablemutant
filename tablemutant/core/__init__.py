#!/usr/bin/env python3
"""
TableMutant - Main orchestrator class for generating new columns
"""

import sys
import logging
from typing import Optional
import polars as pl

from .model_manager import ModelManager
from .table_processor import TableProcessor, HeaderProcessor
from .column_generator import ColumnGenerator
from .rag_processor import RAGProcessor
from .embedding_cache import EmbeddingCache
from .tls_config import get_http_client

# Get logger for this module
logger = logging.getLogger('tablemutant.core')

__all__ = [
    'TableMutant',
    'ModelManager',
    'TableProcessor',
    'HeaderProcessor',
    'ColumnGenerator',
    'RAGProcessor',
    'EmbeddingCache'
]

class TableMutant:
    def __init__(self, settings_manager=None):
        self.settings_manager = settings_manager
        self.model_manager = ModelManager()
        self.table_processor = TableProcessor()
        self.column_generator = ColumnGenerator()
        self.rag_processor = RAGProcessor()
        
    def run(self, args):
        """Main execution flow for CLI mode."""
        # Validate GGUF
        if not self.model_manager.validate_gguf(args.model):
            logger.error("Error: %s is not a valid GGUF model identifier", args.model)
            sys.exit(1)
        
        # Download model if needed
        self.model_manager.model_path = self.model_manager.download_model(args.model)
        logger.info("Using model: %s", self.model_manager.model_path)
        
        # Load table data
        logger.info("Loading table from %s", args.table)
        df = self.table_processor.load_table(args.table)
        
        logger.info("Loaded %s rows with columns: %s", len(df), df.columns)
        
        # Parse column indices first to determine which columns to check for non-empty rows
        column_indices = self.table_processor.parse_column_indices(args.columns, len(df.columns))
        source_columns = [df.columns[i] for i in column_indices]
        logger.info("Using source columns: %s", source_columns)
        
        # Count non-empty rows in the selected columns
        non_empty_count = self.table_processor.count_non_empty_rows(df, column_indices)
        logger.info("Found %s non-empty rows in selected columns", non_empty_count)
        
        # Limit rows if specified, but respect non-empty row count
        num_rows = args.rows if args.rows is not None else non_empty_count
        if num_rows > non_empty_count:
            num_rows = non_empty_count
            logger.info("Limiting to %s non-empty rows", num_rows)
        elif args.rows is not None:
            logger.info("Processing first %s non-empty rows", num_rows)
        
        # Get only non-empty rows for processing
        df_to_process = self.get_non_empty_rows(df, column_indices, num_rows)
        
        
        # Load RAG source if provided
        rag_text = None
        if args.rag_source:
            logger.info("Loading RAG source from %s", args.rag_source)
            
            # Try to use cached embeddings first
            if self.rag_processor.has_cached_embeddings(args.rag_source):
                logger.info("Using cached embeddings...")
                cached_result = self.rag_processor.get_cached_embeddings(args.rag_source)
                if cached_result:
                    text_chunks, embeddings, metadata = cached_result
                    logger.info("Loaded %s cached text chunks", len(text_chunks))
                    # For CLI mode, combine chunks into single text
                    rag_text = "\n\n".join(text_chunks)
                else:
                    # Fallback to raw text loading
                    rag_text = self.rag_processor.load_rag_source(args.rag_source)
            else:
                # Generate embeddings for future use
                logger.info("Generating embeddings for RAG document...")
                result = self.rag_processor.generate_embeddings(args.rag_source)
                if result:
                    text_chunks, embeddings = result
                    logger.info("Generated embeddings for %s text chunks", len(text_chunks))
                    # Use the chunked text
                    rag_text = "\n\n".join(text_chunks)
                else:
                    # Fallback to raw text loading
                    logger.warning("Embedding generation failed, using raw text")
                    rag_text = self.rag_processor.load_rag_source(args.rag_source)
        
        # Download and start llamafile
        self.model_manager.llamafile_path = self.model_manager.download_llamafile()
        self.model_manager.start_llamafile(self.model_manager.model_path)
        
        # Setup DSPy
        self.column_generator.setup_dspy()
        
        # Prepare RAG embeddings data for semantic similarity search
        rag_embeddings_data = None
        if args.rag_source:
            # Check if we have cached embeddings
            if self.rag_processor.has_cached_embeddings(args.rag_source):
                cached_result = self.rag_processor.get_cached_embeddings(args.rag_source)
                if cached_result:
                    text_chunks, embeddings, metadata = cached_result
                    rag_embeddings_data = [{
                        'path': args.rag_source,
                        'chunks': text_chunks,
                        'embeddings': embeddings
                    }]
            else:
                # Generate embeddings if not cached
                result = self.rag_processor.generate_embeddings(args.rag_source)
                if result:
                    text_chunks, embeddings = result
                    rag_embeddings_data = [{
                        'path': args.rag_source,
                        'chunks': text_chunks,
                        'embeddings': embeddings
                    }]
        
        # Generate new column
        new_column_name = args.output_column or "generated_column"
        new_values = self.column_generator.generate_column(
            df_to_process,
            source_columns,
            args.instructions,
            rag_text,
            rag_embeddings_data=rag_embeddings_data,
            rag_processor=self.rag_processor if rag_embeddings_data else None
        )
        
        # Add new column to the processed portion
        df_to_process = df_to_process.with_columns(pl.Series(name=new_column_name, values=new_values))
        
        # If we processed a subset, add empty values for the remaining rows
        if num_rows < len(df):
            # Create a full dataframe with the new column
            remaining_rows = len(df) - num_rows
            empty_values = [""] * remaining_rows
            
            # Get the original remaining rows
            df_remaining = df.tail(remaining_rows)
            df_remaining = df_remaining.with_columns(pl.Series(name=new_column_name, values=empty_values))
            
            # Combine the processed and remaining portions
            df_final = pl.concat([df_to_process, df_remaining])
        else:
            df_final = df_to_process
        
        # Save output
        output_path = args.output
        if output_path:
            # For file output, save the complete table with all rows
            # Adjust default output path
            if not any(output_path.endswith(ext) for ext in ['.csv', '.parquet', '.json']):
                output_path = args.table.rsplit('.', 1)[0] + '_mutated.' + args.table.rsplit('.', 1)[1]
            self.table_processor.save_table(df_final, output_path, new_column_name)
            logger.info("Saved mutated table to %s", output_path)
        else:
            # For stdout, only output the processed rows
            self.table_processor.save_table(df_to_process, output_path, new_column_name)
        
        # Cleanup
        self.cleanup()
    
    def setup_for_gui(self):
        """Setup the model for GUI mode using settings."""
        if not self.settings_manager:
            raise ValueError("SettingsManager is required for GUI mode")
            
        model_identifier = self.settings_manager.get('model')
        if not model_identifier:
            raise ValueError("No model configured in settings")
        
        # Validate GGUF
        if not self.model_manager.validate_gguf(model_identifier):
            raise ValueError(f"{model_identifier} is not a valid GGUF model identifier")
        
        # Download model if needed
        self.model_manager.model_path = self.model_manager.download_model(model_identifier)
        
        # Download and start llamafile with settings
        if not self.model_manager.llamafile_path:
            self.model_manager.llamafile_path = self.model_manager.download_llamafile()
        
        # Determine server host and whether to start local llamafile
        server_host = self.settings_manager.get('server_host')
        auth_token = self.settings_manager.get('auth_token')
        temperature = self.settings_manager.get('temperature')
        max_tokens = self.settings_manager.get('max_tokens')

        def _is_local(url: str) -> bool:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = (parsed.hostname or '').lower()
                return host in ('localhost', '127.0.0.1', '::1')
            except Exception:
                return True  # default to local

        def _extract_port(url: str, default_port: int = 8000) -> int:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                if parsed.port:
                    return int(parsed.port)
                # default ports for schemes if not specified
                if parsed.scheme == 'https':
                    return 443
                if parsed.scheme == 'http':
                    return 80
            except Exception:
                pass
            return default_port

        if _is_local(server_host):
            # Start local llamafile on the specified/derived port
            port = _extract_port(server_host, 8000)
            self.model_manager.start_llamafile(self.model_manager.model_path, port=port)
        # else: remote server; do not start llamafile

        # Setup DSPy pointing at server_host (local or remote)
        self.column_generator.setup_dspy(
            temperature=temperature,
            max_tokens=max_tokens,
            server_host=server_host,
            api_key=auth_token
        )
        
        return True
    
    def setup_model_and_server_only(self):
        """Setup model and server without DSPy configuration (for thread pool execution)."""
        if not self.settings_manager:
            raise ValueError("SettingsManager is required for GUI mode")
            
        model_identifier = self.settings_manager.get('model')
        if not model_identifier:
            raise ValueError("No model configured in settings")
        
        # Validate GGUF
        if not self.model_manager.validate_gguf(model_identifier):
            raise ValueError(f"{model_identifier} is not a valid GGUF model identifier")
        
        # Download model if needed
        self.model_manager.model_path = self.model_manager.download_model(model_identifier)
        
        # Download and start llamafile with settings
        if not self.model_manager.llamafile_path:
            self.model_manager.llamafile_path = self.model_manager.download_llamafile()
        
        # Determine server host and whether to start local llamafile
        server_host = self.settings_manager.get('server_host')

        def _is_local(url: str) -> bool:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = (parsed.hostname or '').lower()
                result = host in ('localhost', '127.0.0.1', '::1')
                logger.debug("_is_local check for %s -> host: %s, result: %s", url, host, result)
                return result
            except Exception as e:
                logger.debug("_is_local failed with error: %s", e)
                return True  # default to local

        def _extract_port(url: str, default_port: int = 8000) -> int:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                if parsed.port:
                    result = int(parsed.port)
                elif parsed.scheme == 'https':
                    result = 443
                elif parsed.scheme == 'http':
                    result = 80
                else:
                    result = default_port
                logger.debug("_extract_port check for %s -> result: %s", url, result)
                return result
            except Exception as e:
                logger.debug("_extract_port failed with error: %s", e)
                return default_port

        if _is_local(server_host):
            port = _extract_port(server_host, 8000)
            logger.debug("Starting llamafile on port %s with model %s", port, self.model_manager.model_path)
            try:
                self.model_manager.start_llamafile(self.model_manager.model_path, port=port)
                logger.debug("Llamafile started successfully on port %s", port)
            except Exception as e:
                logger.debug("Failed to start llamafile: %s", e)
                import traceback
                traceback.print_exc()
                raise
        # else remote: do not start

        return True
    
    def setup_dspy_configuration(self):
        """Setup DSPy configuration (must be called from main thread)."""
        temperature = self.settings_manager.get('temperature')
        max_tokens = self.settings_manager.get('max_tokens')
        server_host = self.settings_manager.get('server_host')
        auth_token = self.settings_manager.get('auth_token')
        logger.debug("Setting up DSPy with server_host: %s", server_host)
        try:
            self.column_generator.setup_dspy(
                temperature=temperature,
                max_tokens=max_tokens,
                server_host=server_host,
                api_key=auth_token
            )
            logger.debug("DSPy setup completed successfully")
        except Exception as e:
            logger.debug("Failed to setup DSPy: %s", e)
            import traceback
            traceback.print_exc()
            raise
        
        return True
        
    def get_non_empty_rows(self, df, column_indices, max_rows=None):
        """Get rows that have at least one non-empty value in the selected columns."""
        if df is None or df.is_empty() or not column_indices:
            return df.head(0) if df is not None else None
        
        # Get column names for the selected indices
        column_names = [df.columns[i] for i in column_indices if i < len(df.columns)]
        if not column_names:
            return df.head(0)
        
        non_empty_rows = []
        for i in range(len(df)):
            has_non_empty = False
            for col_name in column_names:
                try:
                    value = df[col_name][i]
                    # Check if value is non-empty (not None, not empty string, not just whitespace)
                    if value is not None:
                        str_value = str(value).strip()
                        if str_value and str_value.lower() not in ['null', 'na', 'n/a', '']:
                            has_non_empty = True
                            break
                except Exception:
                    continue
            
            if has_non_empty:
                non_empty_rows.append(i)
                # Stop if we've reached the max rows limit
                if max_rows is not None and len(non_empty_rows) >= max_rows:
                    break
        
        # Return the filtered DataFrame with only non-empty rows
        if non_empty_rows:
            return df[non_empty_rows]
        else:
            return df.head(0)  # Return empty DataFrame with same schema
    
    def cleanup(self):
        """Clean up resources."""
        self.model_manager.cleanup()