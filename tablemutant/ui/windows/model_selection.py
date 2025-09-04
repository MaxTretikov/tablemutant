#!/usr/bin/env python3
"""
ModelSelectionWindow - Window for selecting specific model variants to download
"""

import aiohttp
import asyncio
import os
import threading
import time
import traceback
import urllib.parse
import logging
from pathlib import Path

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from tablemutant.core.tls_config import get_async_http_client

# Get logger for this module
logger = logging.getLogger('tablemutant.ui.windows.model_selection')


class ModelSelectionWindow:
    def __init__(self, app, available_files, model_identifier, parent_callback):
        self.app = app
        self.available_files = available_files  # List of tuples: (filename, size_mb, is_grouped_or_parts)
        self.model_identifier = model_identifier
        self.parent_callback = parent_callback
        self.window = None
        self.selection_list = None
        self.selected_file = None
        self.download_section = None
        self.download_status = None
        self.download_progress = None
        self.download_button = None
        self.cancel_button = None
        
    def show(self):
        """Show the model selection window."""
        self.window = toga.Window(title=f"Model Selection - {self.model_identifier}")
        self.window.on_close = self.on_close
        
        content_box = toga.Box(style=Pack(direction=COLUMN, margin=15))
        
        # Title
        title = toga.Label(
            f"Model Selection for {self.model_identifier}",
            style=Pack(margin=(0, 0, 15, 0), font_size=14, font_weight='bold')
        )
        content_box.add(title)
        
        # Info text
        info_text = toga.Label(
            "Select a model variant. Downloaded versions are marked with [downloaded]. Smaller models are faster but less capable.",
            style=Pack(margin=(0, 0, 10, 0), font_size=11, color='#666666')
        )
        content_box.add(info_text)
        
        # Create selection list data
        selection_data = []
        local_files = self._get_local_files()
        
        for filename, size_mb, is_grouped_or_parts in self.available_files:
            if size_mb is not None:
                size_str = f" ({size_mb:.1f} MB)" if size_mb < 1024 else f" ({size_mb/1024:.1f} GB)"
            else:
                size_str = " (size unknown)"
            
            # Check if file is already downloaded
            is_downloaded = self._is_file_downloaded(filename, is_grouped_or_parts, local_files)
            status_str = " [downloaded]" if is_downloaded else ""
            
            # Add part count for grouped files
            if isinstance(is_grouped_or_parts, list):  # This is a grouped file with parts list
                part_count = len(is_grouped_or_parts)
                part_str = f" [{part_count} parts]"
                selection_data.append(f"{filename}{size_str}{part_str}{status_str}")
            else:
                selection_data.append(f"{filename}{size_str}{status_str}")
        
        # Selection list
        self.selection_list = toga.Selection(
            items=selection_data,
            style=Pack(height=300, width=600, margin=(0, 0, 15, 0)),
            on_change=self.on_selection_change
        )
        content_box.add(self.selection_list)
        
        # Buttons
        button_box = toga.Box(style=Pack(direction=ROW, margin=(10, 0, 0, 0)))
        
        self.cancel_button = toga.Button(
            "Cancel",
            on_press=self.cancel,
            style=Pack(margin=(0, 10, 0, 0))
        )
        
        self.download_button = toga.Button(
            "Download Selected",
            on_press=self.download_selected,
            style=Pack(margin=(0, 0, 0, 0))
        )
        
        button_box.add(toga.Box(style=Pack(flex=1)))  # Spacer
        button_box.add(self.cancel_button)
        button_box.add(self.download_button)
        
        content_box.add(button_box)
        
        # Download progress (initially hidden)
        self.download_section = toga.Box(style=Pack(direction=COLUMN, margin=(10, 0, 0, 0)))
        self.download_status = toga.Label(
            "",
            style=Pack(margin=(0, 0, 5, 0))
        )
        self.download_progress = toga.ProgressBar(
            max=100,
            style=Pack(width=400)
        )
        self.download_section.add(self.download_status)
        self.download_section.add(self.download_progress)
        content_box.add(self.download_section)
        
        # Hide download section initially
        self.download_section.style.visibility = 'hidden'
        
        self.window.content = content_box
        self.window.show()
        
        # Update button text based on initial selection (if any)
        self.update_button_text()
    
    def on_selection_change(self, widget):
        """Handle selection change to update button text."""
        self.update_button_text()
    
    def update_button_text(self):
        """Update the button text based on whether the selected model is downloaded."""
        if self.selection_list.value is None:
            self.download_button.text = "Select a Model"
            self.download_button.enabled = False
            return
        
        # Check if selected model is downloaded
        selected_display_text = self.selection_list.value
        if "[downloaded]" in selected_display_text:
            self.download_button.text = "Use Model"
        else:
            self.download_button.text = "Download Model"
        
        self.download_button.enabled = True
    
    async def download_selected(self, widget):
        """Handle the selected model - either use existing or download new."""
        if self.selection_list.value is None:
            await self.app.main_window.dialog(
                toga.ErrorDialog(
                    title="No Selection",
                    message="Please select a model first."
                )
            )
            return
        
        # Check if selected model is already downloaded
        selected_display_text = self.selection_list.value
        if "[downloaded]" in selected_display_text:
            # Model is already downloaded, use it directly
            filename = selected_display_text.split("(")[0].strip()  # Extract filename before size info
            models_dir = Path(self.app.settings_manager.get('models_directory'))
            
            if '/' in self.model_identifier and not self.model_identifier.startswith('http'):
                parts = self.model_identifier.split('/')
                repo_id = '/'.join(parts[:2])
                local_repo_dir = models_dir / repo_id.replace('/', '_')
                local_path = local_repo_dir / filename.replace('.gguf', '') if not filename.endswith('.gguf') else local_repo_dir / filename
                
                if local_path.suffix != '.gguf':
                    # Find the actual .gguf file
                    gguf_files = list(local_repo_dir.glob(f"{filename}*.gguf"))
                    if gguf_files:
                        local_path = gguf_files[0]
            else:
                local_path = models_dir / filename
            
            if local_path.exists():
                await self.app.main_window.dialog(
                    toga.InfoDialog(
                        title="Using Existing Model",
                        message="Using the existing downloaded model."
                    )
                )
                
                if self.parent_callback:
                    await self.parent_callback(str(local_path))
                
                self.window.close()
                self.window = None
                return
            else:
                # File doesn't exist, treat as download
                await self.app.main_window.dialog(
                    toga.ErrorDialog(
                        title="File Not Found",
                        message="The downloaded model file could not be found. It will be re-downloaded."
                    )
                )
        
        # Get selected index and filename/parts
        # The selected value includes metadata like size, so we need to match it properly
        selected_display_text = self.selection_list.value
        selected_index = None
        
        # Try to find exact match first
        try:
            selected_index = self.selection_list.items.index(selected_display_text)
        except ValueError:
            # If exact match fails, find by matching the display text we created
            for i, (filename, size_mb, is_grouped_or_parts) in enumerate(self.available_files):
                # Recreate the display text for this item
                if size_mb is not None:
                    size_str = f" ({size_mb:.1f} MB)" if size_mb < 1024 else f" ({size_mb/1024:.1f} GB)"
                else:
                    size_str = " (size unknown)"
                
                if isinstance(is_grouped_or_parts, list):
                    part_count = len(is_grouped_or_parts)
                    part_str = f" [{part_count} parts]"
                    display_text = f"{filename}{size_str}{part_str}"
                else:
                    display_text = f"{filename}{size_str}"
                
                # Check both with and without [downloaded] status
                if display_text == selected_display_text or f"{display_text} [downloaded]" == selected_display_text:
                    selected_index = i
                    break
        
        if selected_index is None:
            await self.app.main_window.dialog(
                toga.ErrorDialog(
                    title="Selection Error",
                    message="Could not identify the selected model. Please try again."
                )
            )
            return
            
        selected_filename, size_mb, is_grouped_or_parts = self.available_files[selected_index]
        
        # Start download directly in this window
        if isinstance(is_grouped_or_parts, list):
            # This is a grouped file, download the list of parts
            await self.download_model(self.model_identifier, is_grouped_or_parts)
        else:
            # This is a single file
            await self.download_model(self.model_identifier, selected_filename)
    
    async def cancel(self, widget):
        """Cancel model selection."""
        self.window.close()
        self.window = None
    
    async def on_close(self, widget):
        """Handle window close."""
        self.window = None
        return True
    
    async def download_model(self, model_identifier, filename_or_parts):
        """
        Download the model with real-time progress tracking.
        """
        self.download_status.text = f"Downloading {model_identifier}..."
        self.download_section.style.visibility = 'visible'
        self.download_button.enabled = False
        self.cancel_button.enabled = False
        self.download_progress.value = 0
        
        try:
            models_dir = Path(self.app.settings_manager.get('models_directory'))
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle HuggingFace models
            if '/' in model_identifier and not model_identifier.startswith('http'):
                await self._download_huggingface_model(model_identifier, filename_or_parts, models_dir)
            
            # Handle direct URLs
            elif model_identifier.startswith('http'):
                await self._download_url_model(model_identifier, models_dir)
            
            self.download_status.text = "Download complete!"
            self.download_progress.value = 100
            
            # Determine the local path for the downloaded model
            models_dir = Path(self.app.settings_manager.get('models_directory'))
            
            if '/' in model_identifier and not model_identifier.startswith('http'):
                # HuggingFace model
                parts = model_identifier.split('/')
                repo_id = '/'.join(parts[:2])
                local_repo_dir = models_dir / repo_id.replace('/', '_')
                
                # Find the downloaded GGUF file(s)
                if isinstance(filename_or_parts, list):
                    # Multi-part file, use the first part's path
                    local_path = local_repo_dir / filename_or_parts[0]
                else:
                    # Single file
                    local_path = local_repo_dir / filename_or_parts
            else:
                # Direct URL model
                parsed = urllib.parse.urlparse(model_identifier)
                filename = os.path.basename(parsed.path)
                local_path = models_dir / filename
            
            # Verify the file actually exists
            if not local_path.exists():
                logger.warning("Expected downloaded file not found at %s", local_path)
                # Fallback: try to find any GGUF file in the directory
                if local_path.parent.exists():
                    gguf_files = list(local_path.parent.glob('*.gguf'))
                    if gguf_files:
                        local_path = gguf_files[0]  # Use the first GGUF file found
            
            await self.app.main_window.dialog(
                toga.InfoDialog(
                    title="Success",
                    message="Model downloaded successfully!"
                )
            )
            
            # Call parent callback with the local path to update settings
            if self.parent_callback and local_path.exists():
                await self.parent_callback(str(local_path))
            
            # Close window after successful download
            self.window.close()
            self.window = None
            
        except Exception as e:
            # Create a more informative error message
            error_msg = str(e).strip()
            if not error_msg:
                error_msg = f"Unknown error of type {type(e).__name__}"
            
            # Add context about what was being downloaded
            full_error_msg = f"Download failed: {error_msg}"
            if hasattr(e, '__cause__') and e.__cause__:
                full_error_msg += f" (Caused by: {e.__cause__})"
            
            # Log traceback for debugging
            logger.error("Download error for %s:", model_identifier)
            logger.error("%s", traceback.format_exc())
            
            self.download_status.text = full_error_msg
            self.download_button.enabled = True
            self.cancel_button.enabled = True
            
            await self.app.main_window.dialog(
                toga.ErrorDialog(
                    title="Download Error",
                    message=f"Failed to download model '{model_identifier}':\n\n{error_msg}\n\n" + 
                            ("This appears to be a timeout error. Large models can take a long time to download.\n\n"
                             "Suggestions:\n"
                             "• Try downloading at a different time when internet is faster\n"
                             "• Choose a smaller model variant if available\n"
                             "• Check that your internet connection is stable"
                             if "timeout" in error_msg.lower() or "TimeoutError" in error_msg else
                             "Please check your internet connection and try again.")
                )
            )
        
        finally:
            if self.window:  # Only if window still exists
                self.download_button.enabled = True
                self.cancel_button.enabled = True
    
    async def _download_huggingface_model(self, model_identifier, filename_or_parts, models_dir):
        """Download a model from HuggingFace with proper progress tracking."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files, get_hf_file_metadata
            
            parts = model_identifier.split('/')
            repo_id = '/'.join(parts[:2])
            
            # Determine what files to download
            if isinstance(filename_or_parts, list):
                # This is a grouped file with multiple parts
                filenames_to_download = filename_or_parts
                self.download_status.text = f"Downloading {len(filenames_to_download)} parts..."
            else:
                # This is a single file
                if not filename_or_parts:
                    # Find first GGUF file in repo
                    files = list_repo_files(repo_id)
                    gguf_files = [f for f in files if f.endswith('.gguf')]
                    if not gguf_files:
                        raise Exception(f"No GGUF files found in repository {repo_id}")
                    filenames_to_download = [gguf_files[0]]
                else:
                    filenames_to_download = [filename_or_parts]
            
            # For HuggingFace downloads, we'll download each file with direct HTTP tracking
            total_files = len(filenames_to_download)
            
            for i, filename in enumerate(filenames_to_download):
                if total_files > 1:
                    self.download_status.text = f"Downloading part {i+1}/{total_files}: {filename}"
                else:
                    self.download_status.text = f"Downloading {filename}..."
                
                # Get the download URL and file size
                try:
                    metadata = get_hf_file_metadata(
                        url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                    )
                    download_url = metadata.download_url
                    total_size = metadata.size or 0
                except:
                    # If metadata fails, use the direct URL
                    download_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                    total_size = 0
                
                # Download with progress tracking
                local_dir = models_dir / repo_id.replace('/', '_')
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / filename
                
                # Configure timeout for large file downloads
                timeout = aiohttp.ClientTimeout(
                    total=None,  # No total timeout
                    connect=30,  # 30 seconds to connect
                    sock_read=300  # 5 minutes between reads (for slow downloads)
                )
                
                # Retry download up to 3 times on timeout/connection errors
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        async_client = get_async_http_client()
                        async with async_client.session(timeout=timeout) as session:
                            if attempt > 0:
                                self.download_status.text = f"Retrying download {attempt+1}/{max_retries}: {filename}"
                                await asyncio.sleep(2)  # Brief pause before retry
                            
                            async with session.get(download_url) as response:
                                if response.status != 200:
                                    raise Exception(f"Failed to download {filename}: HTTP {response.status}")
                                
                                # Get total size from headers if not already known
                                if total_size == 0:
                                    total_size = int(response.headers.get('content-length', 0))
                                
                                downloaded = 0
                                start_time = time.time()
                                last_update_time = start_time
                                
                                with open(local_path, 'wb') as f:
                                    async for chunk in response.content.iter_chunked(8192):
                                        f.write(chunk)
                                        downloaded += len(chunk)
                                        
                                        current_time = time.time()
                                        # Update progress every 0.1 seconds
                                        if current_time - last_update_time >= 0.1:
                                            progress_info = self._calculate_download_progress(
                                                downloaded, total_size, start_time, current_time
                                            )
                                            
                                            # Calculate overall progress including file number
                                            if total_size > 0:
                                                file_progress = (downloaded / total_size) * 100
                                                overall_progress = (i * 100 + file_progress) / total_files
                                                self.download_progress.value = overall_progress
                                            else:
                                                # Estimate progress
                                                self.download_progress.value = ((i + 0.5) * 100) / total_files
                                            
                                            if total_files > 1:
                                                self.download_status.text = f"Part {i+1}/{total_files}: {progress_info}"
                                            else:
                                                self.download_status.text = progress_info
                                            
                                            last_update_time = current_time
                                        
                                        # Allow other coroutines to run
                                        await asyncio.sleep(0)
                        
                        # If we get here, download succeeded
                        break
                        
                    except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as e:
                        if attempt == max_retries - 1:
                            # Last attempt failed
                            if isinstance(e, asyncio.TimeoutError):
                                raise Exception(f"Download timeout after {max_retries} attempts. File may be too large or connection too slow: {filename}")
                            else:
                                raise Exception(f"Network error after {max_retries} attempts: {str(e) or type(e).__name__}")
                        else:
                            # Not the last attempt, will retry
                            logger.warning("Download attempt %s failed for %s: %s", attempt+1, filename, e)
                            continue
            
            self.download_progress.value = 100
            self.download_status.text = "Download complete!"
            
        except ImportError:
            raise Exception("huggingface-hub library not installed. Please install dependencies by running: pip install huggingface-hub")
        except Exception as e:
            # Re-raise with more context about what failed during HuggingFace download
            error_context = f"HuggingFace download error for {model_identifier}"
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                error_context += f" (HTTP {e.response.status_code})"
            raise Exception(f"{error_context}: {str(e) or type(e).__name__}")
    
    async def _download_url_model(self, url, models_dir):
        """Download a model from a direct URL."""
        try:
            parsed = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed.path)
            local_path = models_dir / filename
            
            # Configure timeout for large file downloads
            async_client = get_async_http_client()
            timeout = async_client.get_timeout(
                total=None,  # No total timeout
                connect=30,  # 30 seconds to connect
                sock_read=300  # 5 minutes between reads (for slow downloads)
            )
            
            # Retry download up to 3 times on timeout/connection errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async_client = get_async_http_client()
                    async with async_client.session(timeout=timeout) as session:
                        if attempt > 0:
                            self.download_status.text = f"Retrying download {attempt+1}/{max_retries}: {filename}"
                            await asyncio.sleep(2)  # Brief pause before retry
                        
                        async with session.get(url) as response:
                            if response.status != 200:
                                raise Exception(f"Server returned HTTP {response.status}: {response.reason}")
                            
                            total_size = int(response.headers.get('content-length', 0))
                            downloaded = 0
                            start_time = time.time()
                            last_update_time = start_time
                            
                            with open(local_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    
                                    current_time = time.time()
                                    # Update progress every 0.1 seconds for smoother updates
                                    if current_time - last_update_time >= 0.1:
                                        progress_info = self._calculate_download_progress(
                                            downloaded, total_size, start_time, current_time
                                        )
                                        
                                        if total_size > 0:
                                            progress = (downloaded / total_size) * 100
                                            self.download_progress.value = progress
                                        else:
                                            # Update progress bar even without known total size
                                            self.download_progress.value = min(self.download_progress.value + 0.5, 95)
                                        
                                        self.download_status.text = progress_info
                                        last_update_time = current_time
                                    
                                    # Allow other coroutines to run
                                    await asyncio.sleep(0)
                            
                            self.download_progress.value = 100
                            self.download_status.text = "Download complete!"
                    
                    # If we get here, download succeeded
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        if isinstance(e, asyncio.TimeoutError):
                            raise Exception(f"Download timeout after {max_retries} attempts. File may be too large or connection too slow: {filename}")
                        else:
                            raise Exception(f"Network error after {max_retries} attempts: {str(e) or type(e).__name__}")
                    else:
                        # Not the last attempt, will retry
                        logger.warning("Download attempt %s failed for %s: %s", attempt+1, filename, e)
                        continue
                        
        except Exception as e:
            # Re-raise with more context about what failed during URL download
            error_msg = str(e) or type(e).__name__
            raise Exception(f"URL download error for {url}: {error_msg}")
    
    def _calculate_download_progress(self, downloaded, total_size, start_time, current_time):
        """Calculate and format tqdm-style download progress information."""
        elapsed_time = current_time - start_time
        
        # Calculate download rate (bytes per second)
        if elapsed_time > 0:
            download_rate = downloaded / elapsed_time
        else:
            download_rate = 0
        
        # Format current downloaded size
        downloaded_str = self._format_size(downloaded)
        
        # Format total size (if known)
        if total_size > 0:
            total_str = self._format_size(total_size)
            size_info = f"{downloaded_str}/{total_str}"
            
            # Calculate ETA
            remaining_bytes = total_size - downloaded
            if download_rate > 0:
                eta_seconds = remaining_bytes / download_rate
                eta_str = self._format_time(eta_seconds)
            else:
                eta_str = "??:??"
        else:
            size_info = downloaded_str
            eta_str = "??:??"
        
        # Format elapsed time
        elapsed_str = self._format_time(elapsed_time)
        
        # Format download rate
        rate_str = self._format_rate(download_rate)
        
        # Return complete progress string with all information
        if total_size > 0:
            # Include percentage when we know the total size
            percentage = (downloaded / total_size) * 100
            return f"{size_info} ({percentage:.1f}%) [{elapsed_str}<{eta_str}, {rate_str}]"
        else:
            return f"{size_info} [{elapsed_str}, {rate_str}]"
    
    def _format_size(self, size_bytes):
        """Format size in human-readable format (B, KB, MB, GB, TB)."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        size_index = 0
        size = float(size_bytes)
        
        while size >= 1024.0 and size_index < len(size_names) - 1:
            size /= 1024.0
            size_index += 1
        
        if size_index == 0:
            return f"{int(size)}{size_names[size_index]}"
        else:
            return f"{size:.1f}{size_names[size_index]}"
    
    def _format_rate(self, bytes_per_second):
        """Format download rate in human-readable format."""
        if bytes_per_second == 0:
            return "0B/s"
        
        rate_str = self._format_size(bytes_per_second)
        return f"{rate_str}/s"
    
    def _format_time(self, seconds):
        """Format time in HH:MM:SS or MM:SS format."""
        if seconds < 0:
            return "??:??"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def _get_local_files(self):
        """Get list of locally downloaded files for this model identifier."""
        models_dir = Path(self.app.settings_manager.get('models_directory'))
        local_files = []
        
        # Handle HuggingFace models
        if '/' in self.model_identifier and not self.model_identifier.startswith('http'):
            parts = self.model_identifier.split('/')
            repo_id = '/'.join(parts[:2])
            local_repo_dir = models_dir / repo_id.replace('/', '_')
            
            if local_repo_dir.exists():
                local_files = [f.name for f in local_repo_dir.glob('*.gguf')]
        
        # Handle direct URLs
        elif self.model_identifier.startswith('http'):
            parsed = urllib.parse.urlparse(self.model_identifier)
            filename = os.path.basename(parsed.path)
            local_path = models_dir / filename
            if local_path.exists():
                local_files = [filename]
        
        return local_files
    
    def _is_file_downloaded(self, filename, is_grouped_or_parts, local_files):
        """Check if a specific file or group of files is already downloaded."""
        if isinstance(is_grouped_or_parts, list):
            # Multi-part file - check if all parts are downloaded
            return all(part_file in local_files for part_file in is_grouped_or_parts)
        else:
            # Single file - check if it exists
            return filename in local_files