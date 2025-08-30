#!/usr/bin/env python3
"""
ProcessingWindow - Handles showing progress during generation processing
"""

import asyncio
import json
import os
import threading
import time
import urllib.request
import logging
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import toga
from toga.style import Pack
from toga.style.pack import COLUMN
import dspy
import polars as pl

# Get logger for this module
logger = logging.getLogger('tablemutant.ui.windows.processing')


class ProcessingWindow:
    def __init__(self, app):
        self.app = app
        self.process_box = None
        self.process_status = None
        self.process_progress = None
        self.time_estimate = None
        self.tokens_info = None
        self.prompt_header = None
        self.prompt_display = None
        self.cancel_button = None
        self.processing_cancelled = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.generation_start_time = None
        self.token_history = deque(maxlen=10)
        self.calibration_tokens_per_sec = 0
        
    def create_content(self, preview_only=True):
        """Create and return the processing window content."""
        self.process_box = toga.Box(style=Pack(direction=COLUMN, margin=10))
        
        # Title
        title = toga.Label(
            "Processing..." if not preview_only else "Generating Preview...",
            style=Pack(margin=(0, 0, 20, 0), font_size=16, font_weight='bold')
        )
        
        # Status
        self.process_status = toga.Label(
            "Initializing model...",
            style=Pack(margin=(0, 0, 10, 0))
        )
        
        # Progress bar
        self.process_progress = toga.ProgressBar(
            max=100,
            style=Pack(width=500, margin=(0, 0, 10, 0))
        )
        
        # Time estimate
        self.time_estimate = toga.Label(
            "Estimating time...",
            style=Pack(margin=(0, 0, 10, 0))
        )
        
        # Tokens info
        self.tokens_info = toga.Label(
            "Tokens: 0 | Speed: -- tokens/sec",
            style=Pack(margin=(0, 0, 20, 0))
        )
        
        # Prompt header
        self.prompt_header = toga.Label(
            "Current Prompt:",
            style=Pack(margin=(0, 0, 5, 0), font_size=14, font_weight='bold')
        )
        
        # Prompt display
        self.prompt_display = toga.MultilineTextInput(
            readonly=True,
            placeholder="Current prompt will appear here...",
            style=Pack(width=500, height=200, margin=(0, 0, 20, 0))
        )
        
        # Cancel button
        self.cancel_button = toga.Button(
            "Cancel",
            on_press=self.cancel_processing,
            style=Pack(margin=5)
        )
        
        self.process_box.add(title)
        self.process_box.add(self.process_status)
        self.process_box.add(self.process_progress)
        self.process_box.add(self.time_estimate)
        self.process_box.add(self.tokens_info)
        self.process_box.add(self.prompt_header)
        self.process_box.add(self.prompt_display)
        self.process_box.add(self.cancel_button)
        
        return self.process_box
    
    async def update_progress(self, value, status_text=None, time_text=None, tokens_text=None):
        """Update progress bar and labels in a thread-safe manner."""
        # Update the GUI elements directly since we're in an async context
        self.process_progress.value = value
        if status_text:
            self.process_status.text = status_text
        if time_text:
            self.time_estimate.text = time_text
        if tokens_text:
            self.tokens_info.text = tokens_text
        
        # Give the GUI event loop a chance to process the updates
        await asyncio.sleep(0.01)
    
    async def update_prompt_display(self, prompt_text, header_text=None):
        """Update the prompt display with the current prompt and optional header."""
        if self.prompt_display:
            self.prompt_display.value = prompt_text
        if header_text and self.prompt_header:
            self.prompt_header.text = header_text
        await asyncio.sleep(0.01)
    
    def format_dspy_prompt(self, signature_class, inputs):
        """Format a DSPy prompt for display based on signature and inputs."""
        prompt_lines = []
        
        # Add signature documentation
        if hasattr(signature_class, '__doc__') and signature_class.__doc__:
            prompt_lines.append(f"Task: {signature_class.__doc__.strip()}")
            prompt_lines.append("")
        
        # Add input fields
        for field_name, field in signature_class.__annotations__.items():
            if hasattr(field, 'desc') and field.desc:
                field_desc = field.desc
            else:
                field_desc = field_name
            
            # Check if this is an input field (not output)
            if field_name in inputs:
                value = inputs[field_name]
                display_value = str(value)
                # if isinstance(value, str) and len(value) > 200:
                #     # Truncate very long values for display
                #     display_value = value[:200] + "..."
                # else:
                #     display_value = str(value)
                
                prompt_lines.append(f"{field_desc}:")
                prompt_lines.append(display_value)
                prompt_lines.append("")
        
        # Add expected output format
        output_fields = []
        for field_name, field in signature_class.__annotations__.items():
            if field_name not in inputs:  # This is an output field
                if hasattr(field, 'desc') and field.desc:
                    output_fields.append(f"- {field_name}: {field.desc}")
                else:
                    output_fields.append(f"- {field_name}")
        
        if output_fields:
            prompt_lines.append("Expected Output:")
            prompt_lines.extend(output_fields)
            prompt_lines.append("")
        
        prompt_lines.append("Please generate the requested output based on the input data above.")
        
        return "\n".join(prompt_lines)
    
    async def process_generation(self, definitions=None, preview_only=True):
        """Process the generation task."""
        try:
            # Initialize model using settings
            await self.update_progress(10, "Setting up model...")
            
            loop = asyncio.get_event_loop()

            dspy.configure_cache(
                enable_disk_cache=False,
                enable_memory_cache=False,
                enable_litellm_cache=False
            )
            
            # Check if server is already running at configured host
            server_host = self.app.settings_manager.get('server_host', 'http://localhost:8000')
            logger.debug("Checking server at %s", server_host)
            auth_token = self.app.settings_manager.get('auth_token', '')
            models_url = (server_host.rstrip('/')) + '/v1/models'
            logger.debug("Models URL: %s", models_url)

            # Determine if host is local to customize status text
            def _is_local(url: str) -> bool:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    host = (parsed.hostname or '').lower()
                    return host in ('localhost', '127.0.0.1', '::1')
                except Exception:
                    return True

            req = urllib.request.Request(models_url)
            if not _is_local(server_host) and auth_token:
                req.add_header('Authorization', f'Bearer {auth_token}')
            try:
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        status_txt = "Server already running!" if not _is_local(server_host) else "Llamafile server already running!"
                        await self.update_progress(40, status_txt)
                        self.app.server_was_running = True
                    else:
                        self.app.server_was_running = False
            except Exception as e:
                logger.debug("Server check failed: %s", e)
                self.app.server_was_running = False
            
            # Always configure DSPy in the main thread, even if server was already running
            if self.app.server_was_running:
                await self.update_progress(45, "Configuring DSPy...")
                try:
                    self.app.tm.setup_dspy_configuration()
                    await self.update_progress(50, "DSPy configuration complete!")
                except Exception as e:
                    # If DSPy configuration fails due to thread conflicts, try to reset and reconfigure
                    print(f"Initial DSPy configuration failed: {e}")
                    await self.update_progress(46, "Retrying DSPy configuration...")
                    try:
                        # Create a fresh column generator instance to avoid thread conflicts
                        from tablemutant.core.column_generator import ColumnGenerator
                        self.app.tm.column_generator = ColumnGenerator()
                        self.app.tm.setup_dspy_configuration()
                        await self.update_progress(50, "DSPy configuration complete!")
                    except Exception as e2:
                        raise Exception(f"Failed to configure DSPy: {e2}")
            
            if not self.app.server_was_running:
                # If host is local, we will start llamafile. If remote, skip starting server.
                server_host = self.app.settings_manager.get('server_host', 'http://localhost:8000')
                logger.debug("Processing server_host: %s", server_host)
                
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
                        return True

                server_host = self.app.settings_manager.get('server_host', 'http://localhost:8000')
                if _is_local(server_host):
                    await self.update_progress(20, "Setting up model and local server...")
                    
                    def setup_model_and_server():
                        logger.debug("Starting setup_model_and_server")
                        result = self.app.tm.setup_model_and_server_only()
                        logger.debug("setup_model_and_server completed")
                        return result
                    
                    try:
                        await loop.run_in_executor(self.executor, setup_model_and_server)
                        await self.update_progress(40, "Model and server setup complete!")
                    except Exception as e:
                        logger.debug("setup_model_and_server failed: %s", e)
                        import traceback
                        traceback.print_exc()
                        raise Exception(f"Failed to setup model: {e}")
                else:
                    await self.update_progress(40, "Using remote server endpoint...")

                # Configure DSPy in the main thread to avoid thread conflicts
                await self.update_progress(45, "Configuring DSPy...")
                try:
                    logger.debug("Starting setup_dspy_configuration")
                    self.app.tm.setup_dspy_configuration()
                    logger.debug("setup_dspy_configuration completed")
                    await self.update_progress(50, "DSPy configuration complete!")
                except Exception as e:
                    # If DSPy configuration fails due to thread conflicts, try to reset and reconfigure
                    print(f"Initial DSPy configuration failed: {e}")
                    await self.update_progress(46, "Retrying DSPy configuration...")
                    try:
                        # Create a fresh column generator instance to avoid thread conflicts
                        from tablemutant.core.column_generator import ColumnGenerator
                        logger.debug("Creating fresh ColumnGenerator instance")
                        self.app.tm.column_generator = ColumnGenerator()
                        logger.debug("Starting retry setup_dspy_configuration")
                        self.app.tm.setup_dspy_configuration()
                        logger.debug("Retry setup_dspy_configuration completed")
                        await self.update_progress(50, "DSPy configuration complete!")
                    except Exception as e2:
                        logger.debug("Retry DSPy configuration failed: %s", e2)
                        import traceback
                        traceback.print_exc()
                        raise Exception(f"Failed to configure DSPy: {e2}")
            
            if self.processing_cancelled:
                return
            
            # Process rows - filter to only non-empty rows first
            if preview_only:
                # For preview, get the first N non-empty rows
                df_to_process = self.get_non_empty_rows(self.app.current_df, self.app.preview_rows)
                num_rows = len(df_to_process)
            else:
                # For full processing, get all non-empty rows
                df_to_process = self.get_non_empty_rows(self.app.current_df, None)
                num_rows = len(df_to_process)
            
            # Get source columns
            source_column_names = [self.app.current_df.columns[i] for i in self.app.selected_columns]
            
            # Load RAG documents and prepare for semantic search
            rag_embeddings_data = []  # Store embedding data for semantic search
            
            if self.app.rag_documents:
                await self.update_progress(55, "Loading RAG documents...")
                
                for doc_path in self.app.rag_documents:
                    try:
                        # Try to get cached embeddings first
                        cached_result = await loop.run_in_executor(
                            self.executor,
                            self.app.tm.rag_processor.get_cached_embeddings,
                            doc_path
                        )
                        
                        if cached_result:
                            text_chunks, embeddings, metadata = cached_result
                            print(f"Using cached embeddings for {os.path.basename(doc_path)} ({len(text_chunks)} chunks)")
                            
                            # Store embedding data for semantic search
                            rag_embeddings_data.append({
                                'path': doc_path,
                                'chunks': text_chunks,
                                'embeddings': embeddings,
                                'metadata': metadata
                            })
                        else:
                            # Fallback: load raw text and create simple chunks
                            print(f"No cached embeddings found for {os.path.basename(doc_path)}, using raw text")
                            doc_text = await loop.run_in_executor(
                                self.executor,
                                self.app.tm.rag_processor.load_rag_source,
                                doc_path
                            )
                            if doc_text:
                                # Create simple chunks from raw text
                                simple_chunks = doc_text.split('\n\n')  # Split by paragraphs
                                simple_chunks = [chunk.strip() for chunk in simple_chunks if chunk.strip()]
                                
                                rag_embeddings_data.append({
                                    'path': doc_path,
                                    'chunks': simple_chunks,
                                    'embeddings': None,  # No embeddings available
                                    'metadata': {'fallback': True}
                                })
                                
                    except Exception as e:
                        print(f"Error loading RAG document {doc_path}: {e}")
                    
                # Log embedding usage
                if rag_embeddings_data:
                    total_chunks = sum(len(data['chunks']) for data in rag_embeddings_data)
                    embedded_docs = sum(1 for data in rag_embeddings_data if data['embeddings'] is not None)
                    print(f"Loaded {len(rag_embeddings_data)} documents with {total_chunks} total chunks")
                    print(f"{embedded_docs} documents have embeddings for semantic search")
            
            # Process each definition
            all_new_columns = {}
            total_operations = len(definitions) * num_rows
            completed = 0
            start_time = time.time()
            self.generation_start_time = time.time()
            
            # Do a calibration run to estimate tokens/sec before starting
            await self.update_progress(60, "Calibrating generation speed...")
            self.calibration_tokens_per_sec = await self.calibrate_generation_speed(loop)
            
            # Initial time estimate based on calibration
            if self.calibration_tokens_per_sec > 0:
                # Estimate tokens per operation (rough estimate)
                estimated_tokens_per_op = 150  # Typical for a ChainOfThought response
                estimated_total_tokens = estimated_tokens_per_op * total_operations
                estimated_time = estimated_total_tokens / self.calibration_tokens_per_sec
                initial_eta = timedelta(seconds=int(estimated_time))
                await self.update_progress(
                    60, 
                    "Starting generation...", 
                    f"Estimated time: ~{initial_eta} (based on {self.calibration_tokens_per_sec:.1f} tokens/sec)",
                    self.calculate_tokens_text()
                )
            
            for def_idx, defn in enumerate(definitions):
                if self.processing_cancelled:
                    break
                
                column_name = " - ".join(defn['headers'])
                
                # Generate values
                new_values = []
                for row_idx in range(num_rows):
                    if self.processing_cancelled:
                        break
                    
                    # Update status before generation
                    status = f"Generating column: {column_name} (row {row_idx + 1}/{num_rows})"
                    
                    # Calculate progress
                    progress = 50 + (completed / total_operations) * 50  # 50-100% range
                    
                    # Calculate time estimate
                    elapsed = time.time() - start_time
                    
                    if completed > 0 and elapsed > 0:
                        # Calculate rate based on completed operations
                        operations_per_sec = completed / elapsed
                        remaining_ops = total_operations - completed
                        
                        if operations_per_sec > 0:
                            remaining_time = remaining_ops / operations_per_sec
                            eta = timedelta(seconds=int(remaining_time))
                            time_text = f"Remaining: ~{eta} ({operations_per_sec:.1f} ops/sec)"
                        else:
                            time_text = "Calculating time estimate..."
                    elif self.token_history and self.calibration_tokens_per_sec > 0:
                        # Use calibration data for initial estimate
                        estimated_tokens_per_op = 150  # Rough estimate
                        remaining_ops = total_operations - completed
                        estimated_remaining_tokens = estimated_tokens_per_op * remaining_ops
                        remaining_time = estimated_remaining_tokens / self.calibration_tokens_per_sec
                        eta = timedelta(seconds=int(remaining_time))
                        time_text = f"Remaining: ~{eta} (estimated)"
                    else:
                        time_text = "Calculating time estimate..."
                    
                    # Calculate tokens per second
                    tokens_text = self.calculate_tokens_text()
                    
                    # Debug output
                    logger.debug(
                        "Progress update - completed: %s/%s, elapsed: %.1fs\n"
                        "Token history length: %s, tokens: %s\n"
                        "Time text: %s\n"
                        "Tokens text: %s",
                        completed, total_operations, elapsed,
                        len(self.token_history), self.total_tokens,
                        time_text, tokens_text
                    )
                    
                    # Update UI before starting generation
                    await self.update_progress(progress, status, time_text, tokens_text)
                    
                    # Prepare row data
                    row_data = {}
                    for col in source_column_names:
                        try:
                            value = df_to_process[col][row_idx]
                            if value is None:
                                row_data[col] = "null"
                            elif isinstance(value, bytes):
                                try:
                                    row_data[col] = value.decode('utf-8', errors='replace')
                                except:
                                    row_data[col] = str(value)
                            else:
                                row_data[col] = str(value)
                        except Exception as e:
                            row_data[col] = ""
                    
                    # Generate value in executor thread
                    def generate_value():
                        # Create DSPy signature and generate
                        class ColumnGeneratorSig(dspy.Signature):
                            """Generate a new column value based on source data and instructions."""
                            source_data = dspy.InputField(desc="Source column data from the current row")
                            task_instructions = dspy.InputField(desc="Instructions for generating the new column")
                            context = dspy.InputField(desc="Additional context from RAG documents")
                            example = dspy.InputField(desc="Example showing input and expected output for reference", default="")
                            new_value = dspy.OutputField(desc="The generated value for the new column")
                        
                        # Use semantic search to find relevant context for this specific row
                        # Process each non-empty line within each field for semantic similarity search
                        rag_context = ""
                        if rag_embeddings_data:
                            all_relevant_chunks = []
                            
                            for col_name, col_value in row_data.items():
                                if col_value and col_value.strip() and col_value != "null":
                                    # Split the field value by newlines to get individual lines
                                    lines = col_value.split('\n')
                                    
                                    for line in lines:
                                        line = line.strip()
                                        if line:  # Only process non-empty lines
                                            # Perform semantic similarity search for this line
                                            relevant_chunks = self.app.tm.rag_processor.find_relevant_chunks(
                                                line, rag_embeddings_data, top_k=5
                                            )
                                            if relevant_chunks:
                                                all_relevant_chunks.extend(relevant_chunks)
                            
                            # Remove duplicates while preserving order
                            seen = set()
                            unique_chunks = []
                            for chunk in all_relevant_chunks:
                                if chunk not in seen:
                                    seen.add(chunk)
                                    unique_chunks.append(chunk)
                            
                            if unique_chunks:
                                rag_context = "\n\nRelevant context from documents:\n" + "\n".join(unique_chunks)
                                print(f"Using {len(unique_chunks)} relevant chunks for row {row_idx + 1} (from {len(all_relevant_chunks)} total matches)")
                            else:
                                print(f"No relevant chunks found for row {row_idx + 1}")
                        
                        # Prepare generation inputs
                        generation_inputs = {
                            "source_data": json.dumps(row_data, indent=2, ensure_ascii=False),
                            "task_instructions": defn['instructions'],
                            "context": rag_context,
                            "example": defn.get('example', '') or ""
                        }
                        
                        generator = dspy.ChainOfThought(ColumnGeneratorSig)
                        
                        try:
                            # Time the generation
                            gen_start = time.time()
                            
                            result = generator(**generation_inputs)
                            
                            gen_time = time.time() - gen_start
                            
                            # Extract token usage from DSPy's language model
                            # Check the configured LM directly for token usage
                            tokens_extracted = False
                            if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
                                history = dspy.settings.lm.history
                                if history and len(history) > 0:
                                    last_call = history[-1]
                                    
                                    if isinstance(last_call, dict):
                                        # Try different ways to extract usage
                                        usage = None
                                        if 'usage' in last_call:
                                            usage = last_call['usage']
                                        elif 'response' in last_call and isinstance(last_call['response'], dict):
                                            usage = last_call['response'].get('usage', {})
                                        elif hasattr(last_call, 'usage'):
                                            usage = last_call.usage
                                        
                                        if usage:
                                            prompt_tokens = usage.get('prompt_tokens', 0)
                                            completion_tokens = usage.get('completion_tokens', 0)
                                            self.total_prompt_tokens += prompt_tokens
                                            self.total_completion_tokens += completion_tokens
                                            self.total_tokens = self.total_prompt_tokens + self.total_completion_tokens
                                            tokens_extracted = True
                                            
                                            # Record token rate
                                            if gen_time > 0 and completion_tokens > 0:
                                                tokens_per_sec = completion_tokens / gen_time
                                                self.token_history.append(tokens_per_sec)
                            
                            # Fallback: estimate tokens if we couldn't extract them
                            if not tokens_extracted:
                                # Estimate tokens based on response length and add to totals
                                estimated_completion = max(10, len(str(result.new_value)) // 4)  # Rough estimate: 4 chars per token
                                estimated_prompt = 100  # Rough estimate for prompt tokens
                                self.total_prompt_tokens += estimated_prompt
                                self.total_completion_tokens += estimated_completion
                                self.total_tokens = self.total_prompt_tokens + self.total_completion_tokens
                                
                                if gen_time > 0:
                                    tokens_per_sec = estimated_completion / gen_time
                                    self.token_history.append(tokens_per_sec)
                            
                            return result.new_value
                        except Exception as e:
                            print(f"Generation error: {e}")
                            return f"Error: {str(e)[:50]}..."
                    
                    # Prepare generation inputs for prompt display (using per-line semantic search)
                    display_rag_context = ""
                    if rag_embeddings_data:
                        def get_display_context():
                            all_relevant_chunks = []
                            
                            for col_name, col_value in row_data.items():
                                if col_value and col_value.strip() and col_value != "null":
                                    # Split the field value by newlines to get individual lines
                                    lines = col_value.split('\n')
                                    
                                    for line in lines:
                                        line = line.strip()
                                        if line:  # Only process non-empty lines
                                            # Perform semantic similarity search for this line
                                            relevant_chunks = self.app.tm.rag_processor.find_relevant_chunks(
                                                line, rag_embeddings_data, top_k=5
                                            )
                                            if relevant_chunks:
                                                all_relevant_chunks.extend(relevant_chunks)
                            
                            # Remove duplicates while preserving order
                            seen = set()
                            unique_chunks = []
                            for chunk in all_relevant_chunks:
                                if chunk not in seen:
                                    seen.add(chunk)
                                    unique_chunks.append(chunk)
                            
                            return unique_chunks
                        
                        # Get display context using per-line search
                        relevant_chunks = await loop.run_in_executor(
                            self.executor,
                            get_display_context
                        )
                        
                        if relevant_chunks:
                            display_rag_context = "\n\nRelevant context from documents:\n" + "\n".join(relevant_chunks)
                    
                    generation_inputs = {
                        "source_data": json.dumps(row_data, indent=2, ensure_ascii=False),
                        "task_instructions": defn['instructions'],
                        "context": display_rag_context,
                        "example": defn.get('example', '') or ""
                    }
                    
                    # Create temporary signature class for prompt formatting
                    class ColumnGeneratorSig(dspy.Signature):
                        """Generate a new column value based on source data and instructions."""
                        source_data = dspy.InputField(desc="Source column data from the current row")
                        task_instructions = dspy.InputField(desc="Instructions for generating the new column")
                        context = dspy.InputField(desc="Additional context from RAG documents")
                        example = dspy.InputField(desc="Example showing input and expected output for reference", default="")
                        new_value = dspy.OutputField(desc="The generated value for the new column")
                    
                    # Format and display the generation prompt
                    prompt_text = self.format_dspy_prompt(ColumnGeneratorSig, generation_inputs)
                    await self.update_prompt_display(prompt_text, f"Generation Prompt (Row {row_idx + 1})")
                    
                    # Run generation in thread pool
                    new_value = await loop.run_in_executor(self.executor, generate_value)
                    new_values.append(new_value)
                    
                    # Increment completed counter after generation is done
                    completed += 1
                    
                    # Force a small yield to allow GUI updates
                    await asyncio.sleep(0.001)
                
                all_new_columns[column_name] = new_values
            
            if self.processing_cancelled:
                return
            
            # Show results
            await self.update_progress(100, "Generation complete!")
            await asyncio.sleep(0.5)  # Brief pause to show completion
            
            if preview_only:
                self.app.show_preview_results_window(df_to_process, all_new_columns, definitions)
            else:
                await self.save_results(all_new_columns)
            
        except Exception as e:
            await self.app.main_window.dialog(
                toga.ErrorDialog(
                    title="Error",
                    message=f"Processing failed: {str(e)}"
                )
            )
            self.app.show_output_definition_window()
        finally:
            # Only cleanup if we started the server
            if not self.app.server_was_running:
                self.app.tm.cleanup()
            # Shutdown the executor
            self.executor.shutdown(wait=False)
    
    def get_non_empty_rows(self, df, max_rows=None):
        """Get rows that have at least one non-empty value in the selected columns."""
        if df is None or df.is_empty() or not self.app.selected_columns:
            return df.head(0) if df is not None else None
        
        # Get column names for the selected indices
        column_names = [df.columns[i] for i in self.app.selected_columns if i < len(df.columns)]
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
    
    async def calibrate_generation_speed(self, loop):
        """Do a quick calibration to estimate tokens per second."""
        try:
            async def calibrate():
                logger.debug("Configuring DSPy cache")
                dspy.configure_cache(
                    enable_disk_cache=False,
                    enable_memory_cache=False,
                    enable_litellm_cache=False
                )
                logger.debug("DSPy cache configured")

                # Simple signature for calibration
                class CalibrationSig(dspy.Signature):
                    """Simple calibration task."""
                    input = dspy.InputField(desc="Input text")
                    output = dspy.OutputField(desc="Output text")
                
                # Prepare calibration inputs
                calibration_inputs = {
                    "input": "Say 'calibration complete' to test the model speed."
                }
                
                # Format and display the calibration prompt
                prompt_text = self.format_dspy_prompt(CalibrationSig, calibration_inputs)
                await self.update_prompt_display(prompt_text, "Calibration Prompt")
                
                # Give UI time to update before starting generation
                await asyncio.sleep(0.1)
                
                calibrator = dspy.ChainOfThought(CalibrationSig)
                
                start_time = time.time()
                logger.debug("Starting calibration generation")
                result = calibrator(**calibration_inputs)
                elapsed = time.time() - start_time
                logger.debug("Calibration generation completed in %.2f seconds", elapsed)
                
                # Try to get token count from the response
                tokens_used = 0
                prompt_tokens = 0
                completion_tokens = 0
                
                # Check the configured LM directly
                if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
                    history = dspy.settings.lm.history
                    if history and len(history) > 0:
                        last_call = history[-1]
                        logger.debug("Last call structure: %s", type(last_call))
                        if hasattr(last_call, 'get'):
                            logger.debug("Last call keys: %s", last_call.keys() if hasattr(last_call, 'keys') else 'no keys')
                        
                        if isinstance(last_call, dict):
                            # Try different ways to extract usage
                            usage = None
                            if 'usage' in last_call:
                                usage = last_call['usage']
                            elif 'response' in last_call and isinstance(last_call['response'], dict):
                                usage = last_call['response'].get('usage', {})
                            elif hasattr(last_call, 'usage'):
                                usage = last_call.usage
                            
                            if usage:
                                prompt_tokens = usage.get('prompt_tokens', 0)
                                completion_tokens = usage.get('completion_tokens', 0)
                                tokens_used = completion_tokens
                                logger.debug("Found tokens - prompt: %s, completion: %s", prompt_tokens, completion_tokens)
                            else:
                                logger.debug("No usage found in last call")
                        else:
                            logger.debug("Last call is not a dict: %s", last_call)
                
                # Update totals with calibration tokens
                if prompt_tokens > 0 or completion_tokens > 0:
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_tokens = self.total_prompt_tokens + self.total_completion_tokens
                    logger.debug("Updated totals - total: %s, prompt: %s, completion: %s",
                               self.total_tokens, self.total_prompt_tokens, self.total_completion_tokens)
                
                if tokens_used > 0 and elapsed > 0:
                    rate = tokens_used / elapsed
                    logger.debug("Calibration rate: %.1f tokens/sec", rate)
                    # Add the calibration rate to our token history
                    self.token_history.append(rate)
                    return rate
                else:
                    # Fallback: estimate based on time (assume ~50 tokens for calibration response)
                    if elapsed > 0:
                        # Still update totals with estimated tokens
                        estimated_tokens = 50
                        self.total_completion_tokens += estimated_tokens
                        self.total_tokens = self.total_prompt_tokens + self.total_completion_tokens
                        logger.debug("Using estimated tokens: %s", estimated_tokens)
                        rate = estimated_tokens / elapsed
                        # Add the estimated rate to our token history
                        self.token_history.append(rate)
                        return rate
                    return 0
            
            tokens_per_sec = await calibrate()
            return tokens_per_sec
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            import traceback
            traceback.print_exc()
            return 0  # Return 0 if calibration fails
    
    def calculate_tokens_text(self):
        """Calculate and format token statistics."""
        logger.debug("calculate_tokens_text - generation_start_time: %s, total_tokens: %s",
                   self.generation_start_time, self.total_tokens)
        
        if self.generation_start_time and self.total_tokens > 0:
            total_time = time.time() - self.generation_start_time
            overall_rate = self.total_tokens / total_time if total_time > 0 else 0
            
            # Calculate average of recent token rates
            if self.token_history:
                recent_rate = sum(self.token_history) / len(self.token_history)
            else:
                recent_rate = 0
            
            result = (f"Tokens: {self.total_tokens:,} total "
                    f"({self.total_prompt_tokens:,} prompt, {self.total_completion_tokens:,} completion) | "
                    f"Speed: {recent_rate:.1f} tokens/sec")
            logger.debug("calculate_tokens_text result: %s", result)
            return result
        else:
            result = "Tokens: 0 | Speed: -- tokens/sec"
            logger.debug("calculate_tokens_text fallback result: %s", result)
            return result
    
    async def cancel_processing(self, widget):
        """Cancel the current processing."""
        self.processing_cancelled = True
        await self.update_progress(0, "Cancelling...")
        self.app.tm.cleanup()
        await asyncio.sleep(1)
        self.app.show_output_definition_window()
    
    async def save_results(self, new_columns):
        """Save the results to file, ensuring preview generations are used at the start."""
        # Get all non-empty rows from the current DataFrame
        all_non_empty_df = self.get_non_empty_rows(self.app.current_df, None)
        
        # Check if we have preview data that should be reused
        preview_rows_count = len(new_columns[list(new_columns.keys())[0]]) if new_columns else 0
        
        if preview_rows_count > 0 and preview_rows_count < len(all_non_empty_df):
            # We have preview data - need to generate for remaining rows
            remaining_rows_df = all_non_empty_df[preview_rows_count:]
            
            # Generate for remaining rows (this would need to be implemented)
            # For now, we'll extend with empty values as a placeholder
            for col_name, preview_values in new_columns.items():
                remaining_count = len(remaining_rows_df)
                # Extend preview values with empty strings for remaining rows
                extended_values = preview_values + [""] * remaining_count
                new_columns[col_name] = extended_values
        
        # Reconstruct the full DataFrame with original headers
        result_df = self.app.original_df.clone()
        
        # Add new columns to the original DataFrame
        for col_name, values in new_columns.items():
            # Pad values to match full DataFrame length
            full_values = []
            
            # Add empty values for header rows
            for _ in range(self.app.header_rows):
                full_values.append("")
            
            # Map generated values to their correct positions in the original DataFrame
            # We need to map from non-empty rows back to original row positions
            non_empty_indices = []
            for i in range(len(self.app.current_df)):
                has_non_empty = False
                for col_idx in self.app.selected_columns:
                    if col_idx < len(self.app.current_df.columns):
                        col_name_check = self.app.current_df.columns[col_idx]
                        try:
                            value = self.app.current_df[col_name_check][i]
                            if value is not None:
                                str_value = str(value).strip()
                                if str_value and str_value.lower() not in ['null', 'na', 'n/a', '']:
                                    has_non_empty = True
                                    break
                        except Exception:
                            continue
                if has_non_empty:
                    non_empty_indices.append(i)
            
            # Create full values array for the working DataFrame
            working_full_values = [""] * len(self.app.current_df)
            for idx, value in enumerate(values):
                if idx < len(non_empty_indices):
                    working_full_values[non_empty_indices[idx]] = value
            
            # Add to the result (after header rows)
            full_values.extend(working_full_values)
            
            # Pad if necessary to match original DataFrame length
            if len(full_values) < len(result_df):
                full_values.extend([""] * (len(result_df) - len(full_values)))
            elif len(full_values) > len(result_df):
                full_values = full_values[:len(result_df)]
            
            # Create a simple column name for the original DataFrame
            new_col_name = f"generated_{len([k for k in new_columns.keys() if k <= col_name])}" if col_name in result_df.columns else col_name
            result_df = result_df.with_columns(pl.Series(name=new_col_name, values=full_values))
        
        # Save file
        output_path = self.app.table_path.rsplit('.', 1)[0] + '_mutated.' + self.app.table_path.rsplit('.', 1)[1]
        self.app.tm.table_processor.save_table(result_df, output_path)
        
        await self.app.main_window.dialog(
            toga.InfoDialog(
                title="Success",
                message=f"Results saved to: {output_path}\n\nPreview generations were used at the start of the file."
            )
        )
        self.app.main_window.close()