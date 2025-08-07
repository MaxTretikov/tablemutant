#!/usr/bin/env python3
"""
SettingsWindow - Application settings configuration
"""

import asyncio
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse

from collections import defaultdict
from pathlib import Path

import aiohttp

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import urllib.parse

from .model_selection import ModelSelectionWindow


class SettingsWindow:
    def __init__(self, app):
        self.app = app
        self.window = None
        self.models_dir = None
        
        # UI elements
        self.model_input = None
        self.model_dropdown = None
        self.check_button = None
        
        # Download progress elements
        self.download_section = None
        self.download_status = None
        self.download_progress = None
        
        # Store original settings
        self.original_settings = None
        
        # Add flag to prevent recursive updates
        self._updating_model_input = False

    def get_models_dir(self):
        """Get the models directory path."""
        if self.models_dir is None:
            self.models_dir = Path(self.app.settings_manager.get('models_directory', 
                                                               Path.home() / '.tablemutant' / 'models'))
        return self.models_dir
        
    def get_local_models(self):
        """Get list of all locally downloaded models."""
        models_dir = self.get_models_dir()
        local_models = []
        
        if models_dir.exists():
            # Find all GGUF files in subdirectories
            for gguf_file in models_dir.rglob('*.gguf'):
                # Create a user-friendly name
                rel_path = gguf_file.relative_to(models_dir)
                if len(rel_path.parts) > 1:
                    # It's in a subdirectory, likely a HuggingFace repo
                    repo_name = rel_path.parts[0].replace('_', '/')
                    filename = rel_path.name
                    display_name = f"{repo_name}/{filename}"
                else:
                    # Direct file
                    display_name = rel_path.name
                
                file_size = gguf_file.stat().st_size if gguf_file.exists() else 0
                size_mb = file_size / (1024 * 1024)
                
                if size_mb < 1024:
                    size_str = f" ({size_mb:.1f} MB)"
                else:
                    size_str = f" ({size_mb/1024:.1f} GB)"
                
                local_models.append({
                    'display_name': f"{display_name}{size_str}",
                    'path': str(gguf_file),
                    'size': file_size
                })
        
        # Sort by display name
        local_models.sort(key=lambda x: x['display_name'])
        return local_models

    def show(self):
        """Show the settings window."""
        if self.window:
            self.window.show()
            return
            
        # Store original settings
        self.original_settings = dict(self.app.settings_manager.settings)
        
        self.window = toga.Window(title="Settings")
        self.window.on_close = self.on_close
        
        # Create content
        content_box = toga.Box(style=Pack(direction=COLUMN, padding=10))
        
        # Title
        title = toga.Label(
            "Application Settings",
            style=Pack(padding=(0, 0, 20, 0), font_size=16, font_weight='bold')
        )
        content_box.add(title)
        
        # Model section
        model_section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 15, 0)))
        model_label = toga.Label(
            "Language Model:",
            style=Pack(padding=(0, 0, 5, 0), font_weight='bold')
        )
        
        model_input_section = toga.Box(style=Pack(direction=ROW))
        
        # Create model input with search autocomplete
        self.model_input = toga.TextInput(
            value=self.app.settings_manager.get('model'),
            placeholder="Type HuggingFace ID, path, or URL",
            style=Pack(flex=1),
            on_change=self.on_model_input_change
        )
        
        # Table for model search results (initially hidden)
        self.model_table_container = toga.Box(style=Pack(direction=COLUMN, padding=(5, 0, 0, 0)))
        self.model_table = None  # Will be created dynamically
        self.current_table_data = []
        self.model_table_lookup = {}
        
        self.check_button = toga.Button(
            "Check",
            on_press=self.check_model,
            style=Pack(padding=(0, 10, 0, 0))
        )

        # Button to fetch remote models when using non-local server
        self.fetch_remote_models_button = toga.Button(
            "Fetch Remote Models",
            on_press=self.fetch_remote_models,
            style=Pack(padding=(0, 0, 0, 10))
        )
        
        model_input_section.add(self.model_input)
        model_input_section.add(self.check_button)
        model_input_section.add(self.fetch_remote_models_button)
        
        model_help = toga.Label(
            "Examples: unsloth/medgemma-27b-text-it-GGUF, /path/to/model.gguf",
            style=Pack(padding=(5, 0, 0, 0), font_size=10, color='#666666')
        )
        
        model_section.add(model_label)
        model_section.add(model_input_section)
        model_section.add(self.model_table_container)  # Search results table
        model_section.add(model_help)

        # Hide remote fetch button for local host by default; toggle later
        def _is_local(url: str) -> bool:
            try:
                parsed = urllib.parse.urlparse(url)
                host = (parsed.hostname or '').lower()
                return host in ('localhost', '127.0.0.1', '::1')
            except Exception:
                return True
        # Will be created later in Server Settings section where host_input exists
        
        # Download progress (initially hidden)
        self.download_section = toga.Box(style=Pack(direction=COLUMN, padding=(10, 0, 0, 0)))
        self.download_status = toga.Label(
            "",
            style=Pack(padding=(0, 0, 5, 0))
        )
        self.download_progress = toga.ProgressBar(
            max=100,
            style=Pack(width=400)
        )
        self.download_section.add(self.download_status)
        self.download_section.add(self.download_progress)
        
        # Server settings section
        server_section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 15, 0)))
        server_label = toga.Label(
            "Server Settings:",
            style=Pack(padding=(0, 0, 10, 0), font_weight='bold')
        )

        # Host input
        host_row = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 5, 0)))
        host_label = toga.Label(
            "Host:",
            style=Pack(width=100)
        )
        self.host_input = toga.TextInput(
            value=self.app.settings_manager.get('server_host', 'http://localhost:8000'),
            placeholder="http://localhost:8000 or https://api.yourserver.com",
            style=Pack(flex=1)
        )
        host_row.add(host_label)
        host_row.add(self.host_input)

        # Auth token (conditionally visible for non-local hosts)
        token_row = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 5, 0)))
        token_label = toga.Label(
            "Auth Token:",
            style=Pack(width=100)
        )
        self.token_input = toga.PasswordInput(
            value=self.app.settings_manager.get('auth_token', ''),
            placeholder="Leave empty if no authentication is required",
            style=Pack(flex=1)
        )
        token_row.add(token_label)
        token_row.add(self.token_input)

        # Helper to toggle token visibility
        def is_local_host(url: str) -> bool:
            try:
                parsed = urllib.parse.urlparse(url)
                host = (parsed.hostname or '').lower()
                return host in ('localhost', '127.0.0.1', '::1')
            except Exception:
                return False

        def toggle_token_visibility(widget=None):
            host_val = self.host_input.value.strip()
            if is_local_host(host_val):
                token_row.style.visibility = 'hidden'
            else:
                token_row.style.visibility = 'visible'

        # Bind change to toggle
        self.host_input.on_change = toggle_token_visibility
        # Initial toggle state
        toggle_token_visibility()

        server_section.add(server_label)
        server_section.add(host_row)
        server_section.add(token_row)

        # Now that host_input exists, finalize toggle that also controls remote fetch button visibility
        def finalize_toggle(widget=None):
            # reuse earlier helper in scope
            def is_local_host(url: str) -> bool:
                try:
                    parsed = urllib.parse.urlparse(url)
                    host = (parsed.hostname or '').lower()
                    return host in ('localhost', '127.0.0.1', '::1')
                except Exception:
                    return True
            host_val = self.host_input.value.strip()
            is_local = is_local_host(host_val)
            token_row.style.visibility = 'hidden' if is_local else 'visible'
            # Show/hide remote fetch button
            self.fetch_remote_models_button.style.visibility = 'hidden' if is_local else 'visible'

        self.host_input.on_change = finalize_toggle
        finalize_toggle()
        
        # Generation settings section
        gen_section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 15, 0)))
        gen_label = toga.Label(
            "Generation Settings:",
            style=Pack(padding=(0, 0, 10, 0), font_weight='bold')
        )
        
        temp_row = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 5, 0)))
        temp_label = toga.Label(
            "Temperature:",
            style=Pack(width=100)
        )
        self.temperature_input = toga.NumberInput(
            value=self.app.settings_manager.get('temperature', 0.7),
            min=0.0,
            max=2.0,
            step=0.1,
            style=Pack(width=100)
        )
        temp_row.add(temp_label)
        temp_row.add(self.temperature_input)
        
        tokens_row = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 5, 0)))
        tokens_label = toga.Label(
            "Max Tokens:",
            style=Pack(width=100)
        )
        self.max_tokens_input = toga.NumberInput(
            value=self.app.settings_manager.get('max_tokens', 2048),
            min=128,
            max=8192,
            step=128,
            style=Pack(width=100)
        )
        tokens_row.add(tokens_label)
        tokens_row.add(self.max_tokens_input)
        
        gen_section.add(gen_label)
        gen_section.add(temp_row)
        gen_section.add(tokens_row)
        
        # Warning label
        warning_label = toga.Label(
            "⚠️ Changing the model will restart the application",
            style=Pack(padding=(10, 0, 10, 0), color='#ff6600', font_weight='bold')
        )
        
        # Buttons
        button_section = toga.Box(style=Pack(direction=ROW, padding=(20, 0, 0, 0)))
        
        # Danger zone buttons (left side)
        danger_buttons = toga.Box(style=Pack(direction=ROW))
        
        delete_models_button = toga.Button(
            "Delete All Models",
            on_press=self.delete_all_models,
            style=Pack(padding=(0, 10, 0, 0), background_color='#dc3545', color='white')
        )
        
        reset_settings_button = toga.Button(
            "Reset to Defaults",
            on_press=self.reset_settings,
            style=Pack(padding=(0, 10, 0, 0), background_color='#6c757d', color='white')
        )
        
        danger_buttons.add(delete_models_button)
        danger_buttons.add(reset_settings_button)
        
        # Main action buttons (right side)
        main_buttons = toga.Box(style=Pack(direction=ROW))
        
        cancel_button = toga.Button(
            "Cancel",
            on_press=self.cancel,
            style=Pack(padding=(0, 10, 0, 0))
        )
        self.apply_button = toga.Button(
            "Apply",
            on_press=self.apply_settings,
            style=Pack(padding=(0, 0, 0, 0))
        )
        
        main_buttons.add(cancel_button)
        main_buttons.add(self.apply_button)
        
        button_section.add(danger_buttons)
        button_section.add(toga.Box(style=Pack(flex=1)))  # Spacer
        button_section.add(main_buttons)
        
        # Add all sections
        content_box.add(model_section)
        content_box.add(server_section)
        content_box.add(gen_section)
        content_box.add(warning_label)
        content_box.add(button_section)
        
        # Add download section last (but don't show it yet)
        content_box.add(self.download_section)
        self.download_section.style.visibility = 'hidden'
        
        self.window.content = content_box
        self.window.show()
    
    async def fetch_remote_models(self, widget):
        """Fetch models from remote OpenAI-compatible /v1/models endpoint and let user choose."""
        try:
            server_host = self.app.settings_manager.get('server_host', 'http://localhost:8000')
            auth_token = self.app.settings_manager.get('auth_token', '')
            # Only allow for non-local hosts
            parsed = urllib.parse.urlparse(server_host)
            host = (parsed.hostname or '').lower() if parsed.hostname else ''
            if host in ('localhost', '127.0.0.1', '::1'):
                await self.app.main_window.dialog(
                    toga.InfoDialog(
                        title="Local Server",
                        message="Remote model fetching is only available for non-local servers."
                    )
                )
                return

            import urllib.request, json
            url = server_host.rstrip('/') + '/v1/models'
            req = urllib.request.Request(url)
            if auth_token:
                req.add_header('Authorization', f'Bearer {auth_token}')
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                data = json.loads(resp.read().decode('utf-8', errors='ignore'))

            # Extract model ids
            models = []
            if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                for item in data['data']:
                    mid = item.get('id') or item.get('name')
                    if mid:
                        models.append(str(mid))
            elif isinstance(data, list):
                # Some servers may return a list
                for item in data:
                    mid = item.get('id') or item.get('name')
                    if mid:
                        models.append(str(mid))

            if not models:
                await self.app.main_window.dialog(
                    toga.InfoDialog(
                        title="No Models",
                        message="No models found at the remote endpoint."
                    )
                )
                return

            # Build a simple selection dialog using a Table
            models = sorted(set(models))
            table = toga.Table(headings=["Model ID"], data=[[m] for m in models], style=Pack(height=200, width=500))
            temp_win = toga.Window(title="Select Remote Model")
            box = toga.Box(style=Pack(direction=COLUMN, padding=10))
            box.add(toga.Label("Select a model from the remote server:", style=Pack(padding=(0,0,10,0))))
            box.add(table)

            async def on_select_and_close(widget_btn):
                if table.selection:
                    # selection is a Row-like; get displayed cell value
                    try:
                        selected_idx = table.data.index(table.selection)
                        selected_model = models[selected_idx]
                    except Exception:
                        # fallback: attempt to read first column
                        selected_model = str(table.selection[0]) if table.selection and len(table.selection) > 0 else None
                    if selected_model:
                        # Set the model_input to the remote model id
                        self._updating_model_input = True
                        self.model_input.value = selected_model
                        self._updating_model_input = False
                temp_win.close()

            btn_row = toga.Box(style=Pack(direction=ROW, padding=(10,0,0,0)))
            btn_ok = toga.Button("Use Selected", on_press=on_select_and_close)
            btn_cancel = toga.Button("Cancel", on_press=lambda w: temp_win.close(), style=Pack(padding=(0,0,0,10)))
            btn_row.add(btn_cancel)
            btn_row.add(btn_ok)
            box.add(btn_row)

            temp_win.content = box
            temp_win.show()

        except Exception as e:
            await self.app.main_window.dialog(
                toga.ErrorDialog(
                    title="Fetch Error",
                    message=f"Failed to fetch remote models: {e}"
                )
            )

    async def check_model(self, widget):
        """Check if the model exists or needs to be downloaded."""
        model_identifier = self.model_input.value.strip()
        if not model_identifier:
            await self.app.main_window.dialog(
                toga.InfoDialog(
                    title="No Model Selected",
                    message="Please enter a model identifier (HuggingFace repo, local path, or URL)."
                )
            )
            return
        
        # Disable check button and update text
        self.check_button.enabled = False
        self.check_button.text = "Checking..."
        
        # Check if it's a HuggingFace repo without specific file
        if '/' in model_identifier and not model_identifier.startswith('http'):
            parts = model_identifier.split('/')
            if len(parts) == 2:  # Only repo, no specific file
                # This MUST use the dropdown - continue with normal flow to show selection window
                pass
            elif len(parts) > 2 and not parts[-1].endswith('.gguf'):
                # Has path but not a GGUF file
                self.check_button.enabled = True
                self.check_button.text = "Check"
                await self.app.main_window.dialog(
                    toga.ErrorDialog(
                        title="Invalid Model Specification",
                        message="When specifying a file in a HuggingFace repo, it must be a .gguf file."
                    )
                )
                return
        
        # Check if it's a local file
        if os.path.exists(model_identifier):
            # Re-enable check button
            self.check_button.enabled = True
            self.check_button.text = "Check"
            
            if model_identifier.endswith('.gguf'):
                await self.app.main_window.dialog(
                    toga.InfoDialog(
                        title="Model Found",
                        message=f"Local model found: {os.path.basename(model_identifier)}"
                    )
                )
            else:
                await self.app.main_window.dialog(
                    toga.ErrorDialog(
                        title="Invalid Model",
                        message="Model file must be in GGUF format"
                    )
                )
            return
        
        # Show download section
        self.download_section.style.visibility = 'visible'
        self.download_status.text = "Checking model availability..."
        self.download_progress.value = 0
        
        try:
            # Check if model exists and find GGUF files
            needs_download, available_files, local_path = await self._check_model_availability(model_identifier)
            
            if not needs_download:
                # Re-enable check button
                self.check_button.enabled = True
                self.check_button.text = "Check"
                
                await self.app.main_window.dialog(
                    toga.InfoDialog(
                        title="Model Available",
                        message=f"Model '{model_identifier}' is already available locally.\n\nPath: {local_path}"
                    )
                )
                self.download_section.style.visibility = 'hidden'
                return
            
            if not available_files:
                # Re-enable check button
                self.check_button.enabled = True
                self.check_button.text = "Check"
                
                await self.app.main_window.dialog(
                    toga.ErrorDialog(
                        title="Model Not Found",
                        message=f"No GGUF files found for '{model_identifier}'"
                    )
                )
                self.download_section.style.visibility = 'hidden'
                return
                
            # Hide download section while showing selection
            self.download_section.style.visibility = 'hidden'
            
            # Get file sizes for better selection
            available_files_with_sizes = await self._get_file_sizes(model_identifier, available_files)
            
            # Re-enable check button after successful check
            self.check_button.enabled = True
            self.check_button.text = "Check"
            
            # Show model selection window
            selection_window = ModelSelectionWindow(
                self.app, 
                available_files_with_sizes, 
                model_identifier, 
                self._on_model_selected
            )
            selection_window.show()
                
        except Exception as e:
            # Re-enable check button on error
            self.check_button.enabled = True
            self.check_button.text = "Check"
            
            await self.app.main_window.dialog(
                toga.ErrorDialog(
                    title="Error",
                    message=f"Error checking model: {str(e)}"
                )
            )
            self.download_section.style.visibility = 'hidden'
    
    async def _on_model_selected(self, model_identifier_or_path, selected_filename_or_parts=None):
        """Callback when a model is selected from the selection window."""
        if selected_filename_or_parts is None:
            # This is a post-download callback with the local path
            local_path = model_identifier_or_path
            
            # Update the model input with the local path using flag to prevent recursive updates
            self._updating_model_input = True
            self.model_input.value = local_path
            self._updating_model_input = False
            
            # Hide the table to prevent automatic selection triggering
            self.hide_model_table()
            
            # Show success message instead of re-checking (we already know it downloaded successfully)
            await self.app.main_window.dialog(
                toga.InfoDialog(
                    title="Model Ready",
                    message=f"Model successfully downloaded and ready to use:\n{os.path.basename(local_path)}"
                )
            )
        else:
            # This is the original pre-download callback
            await self.download_model(model_identifier_or_path, selected_filename_or_parts)
    
    def on_model_input_change(self, widget):
        """Handle model input change to show matching local models in a table.
        
        This provides autocomplete functionality that allows users to:
        - See all local models in a table when input is empty
        - See filtered matches as they type in a dynamically sized table
        - Select from the table by clicking on a row (no auto-fill)
        - Continue editing text after making a selection
        """
        # Prevent recursive updates when we're programmatically setting the value
        if self._updating_model_input:
            return
            
        search_text = widget.value.strip().lower()
        
        # Get all local models
        local_models = self.get_local_models()
        
        if not search_text:
            # Show all models when input is empty
            matching_models = [model['display_name'] for model in local_models]
        else:
            # Search through local models for matches
            matching_models = []
            
            for model in local_models:
                display_name_full = model['display_name'].lower()
                model_path = model['path'].lower()
                
                # Extract base name without size info for searching
                base_name = display_name_full.split('(')[0].strip().lower()
                
                # Check if search text matches any part of the model name or path
                if (search_text in base_name or 
                    search_text in model_path or
                    any(word.startswith(search_text) for word in base_name.split('/')) or
                    any(word.startswith(search_text) for word in base_name.split('-')) or
                    any(word.startswith(search_text) for word in base_name.split('_'))):
                    matching_models.append(model['display_name'])
        
        # Limit results to avoid overwhelming the UI
        matching_models = matching_models[:10]
        
        if matching_models:
            # Show table with matching models
            self.show_model_table(matching_models)
        else:
            # Hide table if no matches (or no local models at all)
            self.hide_model_table()

    def hide_model_table(self):
        """Helper method to hide the model table and clear selection."""
        if self.model_table:
            self.model_table_container.clear()
            self.model_table = None
            self.current_table_data = []
            self.model_table_lookup = {}

    def show_model_table(self, matching_models):
        """Create and show the model table with dynamic sizing."""
        if not matching_models:
            self.hide_model_table()
            return
            
        # Clear existing table
        self.model_table_container.clear()
        
        # Create table data with model info
        table_data = []
        self.model_table_lookup = {}  # Store mapping from table rows to original model data
        local_models = self.get_local_models()
        
        for display_name in matching_models:
            # Find the model details
            for model in local_models:
                if model['display_name'] == display_name:
                    # Extract model name and size separately
                    if '(' in display_name and display_name.endswith(')'):
                        name_part, size_part = display_name.rsplit('(', 1)
                        name = name_part.strip()
                        size = '(' + size_part
                    else:
                        name = display_name
                        size = "—"  # Use a dash for models without size info
                    
                    row_data = [name, size]
                    table_data.append(row_data)
                    # Store mapping from row index to model path
                    self.model_table_lookup[len(table_data) - 1] = model['path']
                    break
        
        # Calculate dynamic height based on number of results (max 6 rows visible for compact display)
        max_visible_rows = min(6, len(table_data))
        row_height = 25  # Approximate height per row
        table_height = max_visible_rows * row_height + 35  # +35 for header and padding
        
        # Create the table
        self.model_table = toga.Table(
            headings=["Model", "Size"],
            data=table_data,
            style=Pack(
                height=table_height,
                flex=1
            ),
            on_select=self.on_model_table_selected
        )
        
        # Store table data for selection handling
        self.current_table_data = table_data
        
        # Add table to container
        self.model_table_container.add(self.model_table)

    async def on_model_table_selected(self, widget):
        """Handle selection of a model from the table.
        
        Uses a lookup table approach to map table row indices to model paths,
        avoiding issues with Row object data access.
        """
        if widget.selection is None or not hasattr(self, 'model_table_lookup'):
            return
            
        # Find the selected row index by comparing with our stored data
        selected_row_index = -1
        for i, row in enumerate(self.current_table_data):
            if widget.selection == widget.data[i]:
                selected_row_index = i
                break
        
        # Use the lookup table to get the model path directly
        if selected_row_index >= 0 and selected_row_index in self.model_table_lookup:
            model_path = self.model_table_lookup[selected_row_index]
            
            # Use flag to prevent recursive updates
            self._updating_model_input = True
            self.model_input.value = model_path
            self._updating_model_input = False
            
            # Hide table
            self.hide_model_table()



    def _detect_multipart_files(self, filenames):
        """Detect and group multi-part files (e.g., 00001-of-00002 patterns)."""
        # Pattern to match files like: filename-00001-of-00002.gguf
        multipart_pattern = re.compile(r'^(.+)-(\d+)-of-(\d+)\.gguf$')
        
        grouped_files = defaultdict(list)
        single_files = []
        
        for filename in filenames:
            match = multipart_pattern.match(filename)
            if match:
                base_name = match.group(1)
                part_num = int(match.group(2))
                total_parts = int(match.group(3))
                grouped_files[base_name].append({
                    'filename': filename,
                    'part_num': part_num,
                    'total_parts': total_parts
                })
            else:
                single_files.append(filename)
        
        # Validate and finalize grouped files
        final_groups = {}
        for base_name, parts in grouped_files.items():
            # Sort parts by part number
            parts.sort(key=lambda x: x['part_num'])
            
            # Validate we have all parts
            total_parts = parts[0]['total_parts']
            if len(parts) == total_parts and all(p['total_parts'] == total_parts for p in parts):
                # Check we have sequential parts starting from 1
                expected_parts = list(range(1, total_parts + 1))
                actual_parts = [p['part_num'] for p in parts]
                if actual_parts == expected_parts:
                    final_groups[base_name] = [p['filename'] for p in parts]
                else:
                    # Missing parts, treat as individual files
                    single_files.extend([p['filename'] for p in parts])
            else:
                # Incomplete set, treat as individual files
                single_files.extend([p['filename'] for p in parts])
        
        return final_groups, single_files
    
    async def _get_file_sizes(self, model_identifier, available_files):
        """Get file sizes for the available files, grouping multi-part files."""
        files_with_sizes = []
        
        # First, detect and group multi-part files
        grouped_files, single_files = self._detect_multipart_files(available_files)
        
        # Handle HuggingFace models
        if '/' in model_identifier and not model_identifier.startswith('http'):
            parts = model_identifier.split('/')
            repo_id = '/'.join(parts[:2])
            
            # Update status to show we're fetching sizes
            self.download_status.text = "Fetching file sizes..."
            
            # Calculate total files to process for progress
            total_files = len(single_files) + sum(len(parts) for parts in grouped_files.values())
            processed_files = 0
            
            async with aiohttp.ClientSession() as session:
                # Process single files
                for filename in single_files:
                    try:
                        # Use HuggingFace resolve URL to get file metadata
                        resolve_url = f'https://huggingface.co/{repo_id}/resolve/main/{filename}'
                        
                        async with session.head(resolve_url) as response:
                            # HuggingFace returns file size in X-Linked-Size header
                            if response.status in [200, 302]:  # 302 is redirect to actual file
                                linked_size = response.headers.get('X-Linked-Size')
                                if linked_size:
                                    size_mb = int(linked_size) / (1024 * 1024)
                                    files_with_sizes.append((filename, size_mb, False))  # False = not grouped
                                else:
                                    # Fallback to content-length
                                    content_length = response.headers.get('content-length')
                                    if content_length:
                                        size_mb = int(content_length) / (1024 * 1024)
                                        files_with_sizes.append((filename, size_mb, False))
                                    else:
                                        files_with_sizes.append((filename, None, False))
                            else:
                                files_with_sizes.append((filename, None, False))
                                
                        processed_files += 1
                        # Update progress while fetching sizes
                        progress = (processed_files / total_files) * 50  # Use 50% of progress bar
                        self.download_progress.value = progress
                        await asyncio.sleep(0.05)  # Allow UI to update
                        
                    except Exception as e:
                        # If we can't get size, just add with None
                        files_with_sizes.append((filename, None, False))
                        print(f"Error getting size for {filename}: {e}")  # Debug info
                        processed_files += 1
                
                # Process grouped files
                for base_name, part_filenames in grouped_files.items():
                    combined_size = 0
                    all_sizes_found = True
                    
                    for filename in part_filenames:
                        try:
                            resolve_url = f'https://huggingface.co/{repo_id}/resolve/main/{filename}'
                            
                            async with session.head(resolve_url) as response:
                                if response.status in [200, 302]:
                                    linked_size = response.headers.get('X-Linked-Size')
                                    if linked_size:
                                        size_mb = int(linked_size) / (1024 * 1024)
                                        combined_size += size_mb
                                    else:
                                        content_length = response.headers.get('content-length')
                                        if content_length:
                                            size_mb = int(content_length) / (1024 * 1024)
                                            combined_size += size_mb
                                        else:
                                            all_sizes_found = False
                                            break
                                else:
                                    all_sizes_found = False
                                    break
                                    
                            processed_files += 1
                            progress = (processed_files / total_files) * 50
                            self.download_progress.value = progress
                            await asyncio.sleep(0.05)
                            
                        except Exception as e:
                            print(f"Error getting size for {filename}: {e}")
                            all_sizes_found = False
                            processed_files += 1
                            break
                    
                    # Add the grouped file entry
                    display_name = f"{base_name}.gguf"
                    total_size = combined_size if all_sizes_found else None
                    # Store the list of part filenames for download
                    files_with_sizes.append((display_name, total_size, part_filenames))
        
        # Handle direct URLs (simpler case, no grouping needed)
        elif model_identifier.startswith('http'):
            filename = available_files[0] if available_files else "model.gguf"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.head(model_identifier) as response:
                        if response.status == 200:
                            content_length = response.headers.get('content-length')
                            if content_length:
                                size_mb = int(content_length) / (1024 * 1024)
                                files_with_sizes.append((filename, size_mb, False))
                            else:
                                files_with_sizes.append((filename, None, False))
                        else:
                            files_with_sizes.append((filename, None, False))
            except Exception:
                files_with_sizes.append((filename, None, False))
        else:
            # Fallback
            files_with_sizes = [(f, None, False) for f in available_files]
        
        # Reset progress and status
        self.download_progress.value = 0
        self.download_status.text = ""
        
        # Sort by size (smaller first, unknowns at the end)
        def sort_key(item):
            filename, size, is_grouped = item
            if size is None:
                return (1, filename)  # Put unknown sizes at end
            return (0, size)
        
        files_with_sizes.sort(key=sort_key)
        return files_with_sizes
    
    async def _check_model_corruption(self, repo_id, local_repo_dir, gguf_files):
        """Check if local model files are corrupted by comparing with remote metadata."""
        corrupted_files = []
        
        try:
            # Get remote file list and metadata
            from huggingface_hub import list_repo_files
            remote_files = list_repo_files(repo_id)
            remote_gguf_files = [f for f in remote_files if f.endswith('.gguf')]
            
            async with aiohttp.ClientSession() as session:
                for local_file in gguf_files:
                    filename = local_file.name
                    
                    # Check if this file exists in the remote repository
                    if filename not in remote_gguf_files:
                        print(f"Local file {filename} not found in remote repository. Marking as corrupted.")
                        corrupted_files.append(local_file)
                        continue
                    
                    # Get remote file size
                    try:
                        resolve_url = f'https://huggingface.co/{repo_id}/resolve/main/{filename}'
                        async with session.head(resolve_url, timeout=10) as response:
                            if response.status in [200, 302]:
                                # Check file size
                                remote_size = None
                                linked_size = response.headers.get('X-Linked-Size')
                                if linked_size:
                                    remote_size = int(linked_size)
                                else:
                                    content_length = response.headers.get('content-length')
                                    if content_length:
                                        remote_size = int(content_length)
                                
                                if remote_size:
                                    local_size = local_file.stat().st_size
                                    if local_size != remote_size:
                                        print(f"Size mismatch for {filename}: local={local_size}, remote={remote_size}. Marking as corrupted.")
                                        corrupted_files.append(local_file)
                                    else:
                                        print(f"✓ {filename} size matches remote ({local_size} bytes)")
                            else:
                                print(f"Could not verify {filename} (HTTP {response.status})")
                    except Exception as e:
                        print(f"Error checking {filename}: {e}")
                        # Don't mark as corrupted if we can't verify - could be network issue
                        
        except Exception as e:
            print(f"Error during corruption check for {repo_id}: {e}")
            # Return empty list if we can't verify - don't delete files due to network errors
            
        return corrupted_files
    
    async def _check_model_availability(self, model_identifier):
        """Check if a model is available and return GGUF files."""
        # Get models directory from settings manager
        models_dir = Path(self.app.settings_manager.get('models_directory', 
                                                       Path.home() / '.tablemutant' / 'models'))
        
        # Handle HuggingFace models
        if '/' in model_identifier and not model_identifier.startswith('http'):
            parts = model_identifier.split('/')
            repo_id = '/'.join(parts[:2])
            
            # Check if already downloaded
            local_repo_dir = models_dir / repo_id.replace('/', '_')
            if local_repo_dir.exists():
                gguf_files = list(local_repo_dir.glob('*.gguf'))
                if gguf_files:
                    # Validate local files for corruption
                    corrupted_files = await self._check_model_corruption(repo_id, local_repo_dir, gguf_files)
                    if corrupted_files:
                        # Delete corrupted files
                        for corrupted_file in corrupted_files:
                            try:
                                corrupted_file.unlink()
                                print(f"Deleted corrupted file: {corrupted_file}")
                            except Exception as e:
                                print(f"Error deleting corrupted file {corrupted_file}: {e}")
                        
                        # Re-check remaining files
                        remaining_files = [f for f in gguf_files if f not in corrupted_files]
                        if not remaining_files:
                            print(f"All local files for {repo_id} were corrupted and deleted. Re-downloading required.")
                        else:
                            return False, [f.name for f in remaining_files], str(local_repo_dir)
                    else:
                        return False, [f.name for f in gguf_files], str(local_repo_dir)
            
            # Check HuggingFace repository
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id)
                gguf_files = [f for f in files if f.endswith('.gguf')]
                return True, gguf_files, None
            except ImportError:
                raise Exception("huggingface-hub not installed. Please install dependencies.")
            except Exception as e:
                raise Exception(f"Error accessing HuggingFace repository: {e}")
        
        # Handle direct URLs
        elif model_identifier.startswith('http'):
            parsed = urllib.parse.urlparse(model_identifier)
            filename = os.path.basename(parsed.path)
            
            if not filename.endswith('.gguf'):
                raise Exception("URL must point to a GGUF file")
            
            local_path = models_dir / filename
            if local_path.exists():
                return False, [filename], str(local_path)
            
            # Check if URL is accessible
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.head(model_identifier) as response:
                        if response.status == 200:
                            return True, [filename], None
                        else:
                            raise Exception(f"URL not accessible (status: {response.status})")
            except Exception as e:
                raise Exception(f"Error checking URL: {e}")
        
        else:
            raise Exception("Invalid model identifier format")
    
    async def download_model(self, model_identifier, filename_or_parts):
        """
        Download the model with real-time progress tracking.
        
        This async method supports:
        - HuggingFace model repositories (e.g., "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
        - Direct GGUF file URLs (e.g., "https://example.com/model.gguf")
        - Progress tracking with visual feedback
        - Error handling with user-friendly dialogs
        """
        self.download_status.text = f"Downloading {model_identifier}..."
        self.apply_button.enabled = False
        
        try:
            models_dir = Path(self.app.settings_manager.get('models_directory', 
                                                           Path.home() / '.tablemutant' / 'models'))
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle HuggingFace models
            if '/' in model_identifier and not model_identifier.startswith('http'):
                await self._download_huggingface_model(model_identifier, filename_or_parts, models_dir)
            
            # Handle direct URLs
            elif model_identifier.startswith('http'):
                await self._download_url_model(model_identifier, models_dir)
            
            self.download_status.text = "Download complete!"
            self.apply_button.enabled = True
            
            await self.app.main_window.dialog(
                toga.InfoDialog(
                    title="Success",
                    message="Model downloaded successfully!"
                )
            )
            
        except Exception as e:
            # Create a more informative error message
            error_msg = str(e).strip()
            if not error_msg:
                error_msg = f"Unknown error of type {type(e).__name__}"
            
            # Add context about what was being downloaded
            full_error_msg = f"Download failed: {error_msg}"
            if hasattr(e, '__cause__') and e.__cause__:
                full_error_msg += f" (Caused by: {e.__cause__})"
            
            # Print traceback to console for debugging
            print(f"Download error for {model_identifier}:")
            print(traceback.format_exc())
            
            self.download_status.text = full_error_msg
            self.apply_button.enabled = True
            
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
            self.download_section.style.visibility = 'hidden'
    
    async def _download_huggingface_model(self, model_identifier, filename_or_parts, models_dir):
        """Download a model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            
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
                    from huggingface_hub import list_repo_files
                    files = list_repo_files(repo_id)
                    gguf_files = [f for f in files if f.endswith('.gguf')]
                    if not gguf_files:
                        raise Exception(f"No GGUF files found in repository {repo_id}")
                    filenames_to_download = [gguf_files[0]]
                else:
                    filenames_to_download = [filename_or_parts]
            
            # Download all files
            import threading
            import time
            
            download_complete = False
            downloaded_paths = []
            download_error = None
            total_files = len(filenames_to_download)
            completed_files = 0
            
            def download_thread():
                nonlocal download_complete, downloaded_paths, download_error, completed_files
                try:
                    for i, filename in enumerate(filenames_to_download):
                        self.download_status.text = f"Downloading part {i+1}/{total_files}: {filename}"
                        
                        local_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            cache_dir=models_dir,
                            local_dir=models_dir / repo_id.replace('/', '_')
                        )
                        downloaded_paths.append(local_path)
                        completed_files += 1
                        
                    download_complete = True
                except Exception as e:
                    download_error = e
                    download_complete = True
            
            # Start download in background thread
            thread = threading.Thread(target=download_thread)
            thread.start()
            
            # Monitor progress
            while not download_complete:
                # Update progress bar with file completion progress + estimated per-file progress
                file_progress = (completed_files / total_files) * 100
                # Add some estimated progress for current file (max 95% until actually complete)
                estimated_current_file_progress = min(file_progress + (5 if completed_files < total_files else 0), 95)
                self.download_progress.value = estimated_current_file_progress
                await asyncio.sleep(0.1)
            
            thread.join()
            
            if download_error:
                raise download_error
                
            self.download_progress.value = 100
            
            if len(downloaded_paths) > 1:
                self.download_status.text = f"Downloaded {len(downloaded_paths)} parts successfully!"
            else:
                self.download_status.text = "Download complete!"
            
        except ImportError:
            raise Exception("huggingface-hub not installed. Please install dependencies.")
    
    async def _download_url_model(self, url, models_dir):
        """Download a model from a direct URL."""
        parsed = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed.path)
        local_path = models_dir / filename
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download: HTTP {response.status}")
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(local_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            self.download_progress.value = progress
                            self.download_status.text = f"Downloaded {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB ({progress:.1f}%)"
                        else:
                            self.download_status.text = f"Downloaded {downloaded / 1024 / 1024:.1f}MB"
                            # Update progress bar even without known total size
                            self.download_progress.value = min(self.download_progress.value + 2, 95)
                        
                        # Allow other coroutines to run
                        await asyncio.sleep(0)
                
                self.download_progress.value = 100

    async def apply_settings(self, widget):
        """Apply settings and restart if model changed."""
        # Normalize host (strip trailing slash)
        host_val = (self.host_input.value or '').strip()
        if host_val.endswith('/'):
            host_val = host_val.rstrip('/')

        new_settings = {
            'model': self.model_input.value.strip(),
            'server_host': host_val or 'http://localhost:8000',
            'auth_token': (self.token_input.value or '').strip(),
            'temperature': float(self.temperature_input.value),
            'max_tokens': int(self.max_tokens_input.value)
        }
        
        # Check if model changed
        model_changed = new_settings['model'] != self.original_settings['model']
        
        # Save settings
        self.app.settings_manager.update(new_settings)
        
        if model_changed:
            # Confirm restart
            result = await self.app.main_window.dialog(
                toga.ConfirmDialog(
                    title="Restart Required",
                    message="The application needs to restart to apply the model change. Continue?"
                )
            )
            
            if result:
                # Close settings window
                self.window.close()
                self.window = None
                
                # Restart the application
                self.restart_application()
            else:
                # Revert model setting
                self._updating_model_input = True
                self.model_input.value = self.original_settings['model']
                self._updating_model_input = False
                self.app.settings_manager.set('model', self.original_settings['model'])
        else:
            # No restart needed, but update file window
            self.update_file_window_model_status()
            await self.app.main_window.dialog(
                toga.InfoDialog(
                    title="Success",
                    message="Settings saved successfully!"
                )
            )
            self.window.close()
            self.window = None
    
    async def cancel(self, widget):
        """Cancel and close the settings window."""
        self.window.close()
        self.window = None
    
    async def on_close(self, widget):
        """Handle window close."""
        self.window = None
        return True
    
    async def delete_all_models(self, widget):
        """Delete all downloaded models from the local models directory."""
        # Get models directory
        models_dir = Path(self.app.settings_manager.get('models_directory', 
                                                       Path.home() / '.tablemutant' / 'models'))
        
        # Check if directory exists and has contents
        if not models_dir.exists():
            await self.app.main_window.dialog(
                toga.InfoDialog(
                    title="No Models Found",
                    message="No models directory found. No models to delete."
                )
            )
            return
        
        # Count files and directories
        try:
            contents = list(models_dir.iterdir())
            if not contents:
                await self.app.main_window.dialog(
                    toga.InfoDialog(
                        title="No Models Found",
                        message="Models directory is empty. No models to delete."
                    )
                )
                return
        except Exception as e:
            await self.app.main_window.dialog(
                toga.ErrorDialog(
                    title="Error",
                    message=f"Could not access models directory: {e}"
                )
            )
            return
        
        # Show confirmation dialog
        result = await self.app.main_window.dialog(
            toga.ConfirmDialog(
                title="Delete All Models",
                message=f"Are you sure you want to delete ALL downloaded models?\n\nThis will remove {len(contents)} items from:\n{models_dir}\n\nThis action cannot be undone."
            )
        )
        
        if result:
            try:
                # Delete all contents of the models directory
                for item in contents:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                
                await self.app.main_window.dialog(
                    toga.InfoDialog(
                        title="Success",
                        message=f"Successfully deleted all models from:\n{models_dir}"
                    )
                )
                
            except Exception as e:
                await self.app.main_window.dialog(
                    toga.ErrorDialog(
                        title="Error",
                        message=f"Failed to delete some models: {e}"
                    )
                )
    
    async def reset_settings(self, widget):
        """Reset all settings to their default values."""
        result = await self.app.main_window.dialog(
            toga.ConfirmDialog(
                title="Reset Settings",
                message="Are you sure you want to reset all settings to their default values?\n\nThis will:\n• Reset the model to the default\n• Reset server host to http://localhost:8000\n• Reset temperature to 0.7\n• Reset max tokens to 2048\n• Reset models directory to default\n\nThe application will restart after resetting."
            )
        )
        
        if result:
            try:
                # Get default settings
                default_settings = self.app.settings_manager.get_default_settings()
                
                # Update the UI inputs with default values
                self._updating_model_input = True
                self.model_input.value = default_settings['model']
                self._updating_model_input = False
                # New server defaults
                self.host_input.value = default_settings.get('server_host', 'http://localhost:8000')
                self.token_input.value = default_settings.get('auth_token', '')
                self.temperature_input.value = default_settings['temperature']
                self.max_tokens_input.value = default_settings['max_tokens']
                
                # Save the default settings
                self.app.settings_manager.update(default_settings)
                
                await self.app.main_window.dialog(
                    toga.InfoDialog(
                        title="Settings Reset",
                        message="Settings have been reset to defaults. The application will now restart."
                    )
                )
                
                # Close settings window
                self.window.close()
                self.window = None
                
                # Restart the application
                self.restart_application()
                
            except Exception as e:
                await self.app.main_window.dialog(
                    toga.ErrorDialog(
                        title="Error",
                        message=f"Failed to reset settings: {e}"
                    )
                )
    
    def restart_application(self):
        """Restart the application."""
        # Save current python executable and script
        python = sys.executable
        script = sys.argv[0]
        args = sys.argv[1:]
        
        # Close the app cleanly
        self.app.tm.cleanup()
        
        # Start new instance and exit
        import subprocess
        subprocess.Popen([python, script] + args)
        
        # Exit current instance
        self.app.exit()
    
    def update_file_window_model_status(self):
        """Update the model status in file selection window if it exists."""
        if self.app.file_window and hasattr(self.app.file_window, 'refresh_model_status'):
            self.app.file_window.refresh_model_status()