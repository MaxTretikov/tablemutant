#!/usr/bin/env python3
"""
EmbeddingProgressWindow - Shows progress during RAG embedding generation as a separate window
"""

import asyncio
import os
import time
from datetime import timedelta
import toga
from toga.style import Pack
from toga.style.pack import COLUMN


class EmbeddingProgressWindow:
    def __init__(self, app):
        self.app = app
        self.window = None
        self.progress_box = None
        self.progress_bar = None
        self.status_label = None
        self.filename_label = None
        self.batch_info_label = None
        self.eta_label = None
        self.speed_label = None
        self.cancel_button = None
        self.cancelled = False
        
        # Progress tracking variables
        self.start_time = None
        self.current_batch = 0
        self.total_batches = 0
        self.batch_times = []
        
    def create_window(self, filename):
        """Create and show the embedding progress window."""
        self.window = toga.Window(
            title="Generating Embeddings",
            size=(500, 300),
            resizable=False
        )
        
        self.progress_box = toga.Box(style=Pack(direction=COLUMN, padding=20))
        
        # Title
        title = toga.Label(
            "Generating Document Embeddings",
            style=Pack(padding=(0, 0, 20, 0), font_size=16, font_weight='bold')
        )
        
        # Filename
        self.filename_label = toga.Label(
            f"Processing: {filename}",
            style=Pack(padding=(0, 0, 15, 0), font_size=12)
        )
        
        # Status
        self.status_label = toga.Label(
            "Initializing...",
            style=Pack(padding=(0, 0, 10, 0))
        )
        
        # Progress bar
        self.progress_bar = toga.ProgressBar(
            max=100,
            style=Pack(width=450, padding=(0, 0, 10, 0))
        )
        
        # Batch information
        self.batch_info_label = toga.Label(
            "Batches: 0/0",
            style=Pack(padding=(0, 0, 5, 0), font_size=10)
        )
        
        # ETA information
        self.eta_label = toga.Label(
            "ETA: Calculating...",
            style=Pack(padding=(0, 0, 5, 0), font_size=10)
        )
        
        # Speed information
        self.speed_label = toga.Label(
            "Speed: -- batches/sec",
            style=Pack(padding=(0, 0, 15, 0), font_size=10)
        )
        
        # Cancel button
        self.cancel_button = toga.Button(
            "Cancel",
            on_press=self.cancel_embedding,
            style=Pack(padding=5)
        )
        
        self.progress_box.add(title)
        self.progress_box.add(self.filename_label)
        self.progress_box.add(self.status_label)
        self.progress_box.add(self.progress_bar)
        self.progress_box.add(self.batch_info_label)
        self.progress_box.add(self.eta_label)
        self.progress_box.add(self.speed_label)
        self.progress_box.add(self.cancel_button)
        
        self.window.content = self.progress_box
        self.window.show()
        
        return self.window
    
    async def update_progress(self, progress_value, status_text=None, current_batch=None, total_batches=None):
        """Update progress bar and status text with tqdm-like information."""
        if self.progress_bar:
            self.progress_bar.value = min(100, max(0, progress_value))
        
        if status_text and self.status_label:
            self.status_label.text = status_text
        
        # Update batch information
        if current_batch is not None and total_batches is not None:
            self.current_batch = current_batch
            self.total_batches = total_batches
            
            if self.batch_info_label:
                self.batch_info_label.text = f"Batches: {current_batch}/{total_batches}"
            
            # Track timing for ETA calculation
            current_time = time.time()
            if self.start_time is None:
                self.start_time = current_time
            
            # Calculate speed and ETA
            if current_batch > 0:
                elapsed_time = current_time - self.start_time
                batches_per_sec = current_batch / elapsed_time if elapsed_time > 0 else 0
                
                if self.speed_label:
                    self.speed_label.text = f"Speed: {batches_per_sec:.2f} batches/sec"
                
                # Calculate ETA
                if batches_per_sec > 0 and current_batch < total_batches:
                    remaining_batches = total_batches - current_batch
                    eta_seconds = remaining_batches / batches_per_sec
                    eta_timedelta = timedelta(seconds=int(eta_seconds))
                    
                    if self.eta_label:
                        self.eta_label.text = f"ETA: {eta_timedelta}"
                elif current_batch >= total_batches:
                    if self.eta_label:
                        self.eta_label.text = "ETA: Complete!"
        
        # Give the GUI event loop a chance to process updates
        await asyncio.sleep(0.01)
    
    async def cancel_embedding(self, widget):
        """Cancel the embedding generation."""
        self.cancelled = True
        await self.update_progress(0, "Cancelling...")
    
    def close_window(self):
        """Close the progress window."""
        if self.window:
            self.window.close()
            self.window = None
    
    async def show_progress_and_generate(self, file_path_str, filename):
        """Show progress dialog and generate embeddings."""
        try:
            # Create progress callback that updates the UI
            async def progress_callback(progress_value, status_text, current_batch=None, total_batches=None):
                if not self.cancelled:
                    await self.update_progress(progress_value, status_text, current_batch, total_batches)
            
            # Run embedding generation in thread pool to avoid blocking UI
            loop = asyncio.get_event_loop()
            from concurrent.futures import ThreadPoolExecutor
            
            # Create a wrapper that handles the progress callback
            def generate_embeddings():
                # Create a synchronous progress callback wrapper
                def sync_progress_callback(progress_value, status_text, current_batch=None, total_batches=None):
                    if not self.cancelled:
                        # Schedule the async update
                        asyncio.run_coroutine_threadsafe(
                            progress_callback(progress_value, status_text, current_batch, total_batches),
                            loop
                        )
                
                return self.app.tm.rag_processor.generate_embeddings(
                    file_path_str,
                    progress_callback=sync_progress_callback
                )
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(executor, generate_embeddings)
            
            if self.cancelled:
                return None
            
            await self.update_progress(100, "Complete!")
            await asyncio.sleep(1)  # Brief pause to show completion
            
            return result
            
        except Exception as e:
            await self.update_progress(0, f"Error: {str(e)}")
            raise e