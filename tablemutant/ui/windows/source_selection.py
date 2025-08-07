#!/usr/bin/env python3
"""
SourceSelectionWindow - Handles selecting source columns and uploading RAG documents
"""

import os
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from toga.sources import Row


class SourceSelectionWindow:
    def __init__(self, app):
        self.app = app
        self.source_box = None
        self.column_dropdown = None
        self.selected_list = None
        self.column_options = None
        self.next_button = None  # Add reference to next button
        
    def create_content(self):
        """Create and return the source selection window content."""
        self.source_box = toga.Box(style=Pack(direction=COLUMN, padding=10))
        
        # Title
        title = toga.Label(
            "Select Data Sources",
            style=Pack(padding=(0, 0, 20, 0), font_size=16, font_weight='bold')
        )
        
        # Table columns section
        columns_section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 15, 0)))
        columns_label = toga.Label(
            "Table Columns:",
            style=Pack(padding=(0, 0, 10, 0), font_weight='bold')
        )
        
        # Column dropdown
        column_dropdown_section = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 10, 0)))
        
        # Prepare column options - now they're already in the DataFrame columns
        self.column_options = {}
        if self.app.current_df is not None:
            for i, col in enumerate(self.app.current_df.columns):
                self.column_options[col] = i
        
        self.column_dropdown = toga.Selection(
            items=list(self.column_options.keys()) if self.column_options else [],
            style=Pack(flex=1)
        )
        
        add_column_button = toga.Button(
            "Add Column",
            on_press=self.add_source_column,
            style=Pack(padding=(0, 10, 0, 0))
        )
        
        column_dropdown_section.add(self.column_dropdown)
        column_dropdown_section.add(add_column_button)
        
        columns_section.add(columns_label)
        columns_section.add(column_dropdown_section)
        
        # Document upload section
        documents_section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 15, 0)))
        documents_label = toga.Label(
            "RAG Documents:",
            style=Pack(padding=(0, 0, 10, 0), font_weight='bold')
        )
        
        add_document_button = toga.Button(
            "Add Document",
            on_press=self.add_rag_document,
            style=Pack(padding=(0, 0, 10, 0))
        )
        
        documents_section.add(documents_label)
        documents_section.add(add_document_button)
        
        # Selected sources list
        list_label = toga.Label(
            "Selected Sources:",
            style=Pack(padding=(10, 0, 5, 0), font_weight='bold')
        )
        
        # Scrollable container for custom list items
        self.list_scroll_container = toga.ScrollContainer(
            style=Pack(flex=1, padding=5)
        )
        
        # Container for list items (will be populated dynamically)
        self.sources_list_box = toga.Box(style=Pack(direction=COLUMN, padding=5))
        self.list_scroll_container.content = self.sources_list_box
        
        # Keep track of source items for easy management
        self.source_items = []
        
        # Navigation buttons
        nav_section = toga.Box(style=Pack(direction=ROW, padding=(20, 0, 0, 0)))
        back_button = toga.Button(
            "Back",
            on_press=self.back_to_file_selection,
            style=Pack(padding=5)
        )
        self.next_button = toga.Button(  # Store reference to next button
            "Next: Define Output Columns",
            on_press=self.confirm_source_selection,
            style=Pack(padding=5),
            enabled=False  # Start disabled
        )
        nav_section.add(back_button)
        nav_section.add(toga.Box(style=Pack(flex=1)))
        nav_section.add(self.next_button)
        
        # Add all components
        self.source_box.add(title)
        self.source_box.add(columns_section)
        self.source_box.add(documents_section)
        self.source_box.add(list_label)
        self.source_box.add(self.list_scroll_container)
        self.source_box.add(nav_section)
        
        # Load existing state from app
        self.load_existing_state()
        
        return self.source_box
    
    def load_existing_state(self):
        """Load existing selected sources from app state and recreate UI items."""
        # Clear any existing UI items first
        self.source_items = []
        for child in list(self.sources_list_box.children):
            self.sources_list_box.remove(child)
        
        # Load selected columns
        if self.app.selected_columns and self.app.current_df is not None:
            for col_idx in self.app.selected_columns:
                if col_idx < len(self.app.current_df.columns):
                    col_name = self.app.current_df.columns[col_idx]
                    self.add_source_item(
                        title=col_name,
                        subtitle=f"Column index: {col_idx}",
                        item_type='column',
                        col_idx=col_idx
                    )
        
        # Load RAG documents
        if self.app.rag_documents:
            for file_path in self.app.rag_documents:
                filename = os.path.basename(file_path)
                
                # Check if embeddings are cached and get chunk count
                if self.app.tm.rag_processor.has_cached_embeddings(file_path):
                    cached_result = self.app.tm.rag_processor.get_cached_embeddings(file_path)
                    if cached_result:
                        text_chunks, embeddings, metadata = cached_result
                        chunk_count = len(text_chunks)
                        subtitle = f"{file_path} (cached, {chunk_count} chunks)"
                    else:
                        subtitle = f"{file_path} (cached, 0 chunks)"
                else:
                    subtitle = f"{file_path} (no embeddings)"
                
                self.add_source_item(
                    title=filename,
                    subtitle=subtitle,
                    item_type='document',
                    file_path=file_path
                )
        
        # Update button state
        self.update_next_button_state()
    
    def update_next_button_state(self):
        """Update the next button enabled state based on whether sources are selected."""
        has_sources = bool(self.app.selected_columns or self.app.rag_documents)
        if self.next_button:
            self.next_button.enabled = has_sources
    
    async def add_source_column(self, widget):
        """Add selected column to source list."""
        if not self.column_dropdown.value:
            return
            
        selected = self.column_dropdown.value
        col_idx = self.column_options[selected]
        
        # Check if already added
        for item in self.source_items:
            if item['type'] == 'column' and item['col_idx'] == col_idx:
                return  # Already exists
        
        self.app.selected_columns.append(col_idx)
        
        # Create custom list item with remove button
        self.add_source_item(
            title=selected,
            subtitle=f"Column index: {col_idx}",
            item_type='column',
            col_idx=col_idx
        )
        
        # Update button state
        self.update_next_button_state()
    
    async def add_rag_document(self, widget):
        """Add a RAG document via file dialog."""
        try:
            file_dialog = toga.OpenFileDialog(
                title="Select Document for RAG",
                file_types=["pdf", "txt", "md"]
            )
            file_path = await self.app.main_window.dialog(file_dialog)
            
            if file_path:
                file_path_str = str(file_path)
                filename = os.path.basename(file_path_str)
                
                # Check if already added
                for item in self.source_items:
                    if item['type'] == 'document' and item['file_path'] == file_path_str:
                        return  # Already exists
                
                try:
                    # Check if embeddings are already cached
                    if self.app.tm.rag_processor.has_cached_embeddings(file_path_str):
                        # Get cached embeddings to show chunk count
                        cached_result = self.app.tm.rag_processor.get_cached_embeddings(file_path_str)
                        if cached_result:
                            text_chunks, embeddings, metadata = cached_result
                            chunk_count = len(text_chunks)
                        else:
                            chunk_count = 0
                        
                        # Embeddings already exist, just add to list
                        self.app.rag_documents.append(file_path_str)
                        
                        # Create custom list item with remove button - show chunk count
                        self.add_source_item(
                            title=filename,
                            subtitle=f"{file_path_str} (cached, {chunk_count} chunks)",
                            item_type='document',
                            file_path=file_path_str
                        )
                    else:
                        # Need to generate embeddings - show progress dialog as separate window
                        from tablemutant.ui.windows.embedding_progress import EmbeddingProgressWindow
                        
                        # Create and show progress window
                        progress_window = EmbeddingProgressWindow(self.app)
                        progress_window.create_window(filename)
                        
                        try:
                            # Generate embeddings with progress
                            result = await progress_window.show_progress_and_generate(file_path_str, filename)
                            
                            # Close progress window
                            progress_window.close_window()
                            
                            if result and not progress_window.cancelled:
                                text_chunks, embeddings = result
                                self.app.rag_documents.append(file_path_str)
                                
                                # Create custom list item with remove button
                                self.add_source_item(
                                    title=filename,
                                    subtitle=f"{file_path_str} ({len(text_chunks)} chunks)",
                                    item_type='document',
                                    file_path=file_path_str
                                )
                                
                                await self.app.main_window.dialog(
                                    toga.InfoDialog(
                                        title="Document Added",
                                        message=f"Document '{filename}' processed successfully.\nGenerated {len(text_chunks)} text chunks and embeddings."
                                    )
                                )
                            elif progress_window.cancelled:
                                # User cancelled, don't add document
                                return
                            else:
                                await self.app.main_window.dialog(
                                    toga.ErrorDialog(
                                        title="Processing Failed",
                                        message=f"Failed to generate embeddings for '{filename}'.\nThe document will be added but may not work optimally for RAG."
                                    )
                                )
                                
                                # Still add the document even if embedding generation failed
                                self.app.rag_documents.append(file_path_str)
                                
                                self.add_source_item(
                                    title=filename,
                                    subtitle=f"{file_path_str} (no embeddings)",
                                    item_type='document',
                                    file_path=file_path_str
                                )
                        except Exception as progress_error:
                            # Close progress window on error
                            progress_window.close_window()
                            raise progress_error
                
                except Exception as embedding_error:
                    print(f"Error during embedding generation: {embedding_error}")
                    await self.app.main_window.dialog(
                        toga.ErrorDialog(
                            title="Processing Error",
                            message=f"Error processing document '{filename}': {str(embedding_error)}\nThe document will be added but may not work optimally."
                        )
                    )
                    
                    # Still add the document even if there was an error
                    if file_path_str not in self.app.rag_documents:
                        self.app.rag_documents.append(file_path_str)
                        
                        self.add_source_item(
                            title=filename,
                            subtitle=f"{file_path_str} (processing error)",
                            item_type='document',
                            file_path=file_path_str
                        )
                
                # Update button state
                self.update_next_button_state()
                
        except Exception as e:
            await self.app.main_window.dialog(
                toga.ErrorDialog(
                    title="Error",
                    message=f"Failed to load document: {str(e)}"
                )
            )
    
    def add_source_item(self, title, subtitle, item_type, col_idx=None, file_path=None):
        """Add a custom source item with a remove button."""
        # Create container for this item
        item_box = toga.Box(style=Pack(direction=ROW, padding=(5, 0, 5, 0)))
        
        # Content box for title and subtitle
        content_box = toga.Box(style=Pack(direction=COLUMN, flex=1, padding=(0, 10, 0, 0)))
        
        title_label = toga.Label(
            title,
            style=Pack(font_weight='bold')
        )
        subtitle_label = toga.Label(
            subtitle,
            style=Pack(font_size=10, color='#666666')
        )
        
        content_box.add(title_label)
        content_box.add(subtitle_label)
        
        # Store item data first
        item_data = {
            'type': item_type,
            'title': title,
            'subtitle': subtitle,
            'col_idx': col_idx,
            'file_path': file_path,
            'box': item_box
        }
        
        # Create remove function that captures the item_data
        def create_remove_handler(data):
            async def handler(widget):
                await self.remove_source_item(widget, data)
            return handler
        
        # Remove button with minus symbol
        remove_button = toga.Button(
            "−",  # Unicode minus symbol
            on_press=create_remove_handler(item_data),
            style=Pack(width=30, height=30, padding=(0, 5, 0, 0), background_color='#ff4444', color='white')
        )
        
        item_box.add(content_box)
        item_box.add(remove_button)
        
        # Store in tracking list
        self.source_items.append(item_data)
        
        # Add to UI
        self.sources_list_box.add(item_box)
    
    async def remove_source_item(self, widget, item_data):
        """Remove a specific source item."""
        # Remove from app data
        if item_data['type'] == 'column' and item_data['col_idx'] is not None:
            if item_data['col_idx'] in self.app.selected_columns:
                self.app.selected_columns.remove(item_data['col_idx'])
        elif item_data['type'] == 'document' and item_data['file_path'] is not None:
            if item_data['file_path'] in self.app.rag_documents:
                self.app.rag_documents.remove(item_data['file_path'])
        
        # Remove from UI
        self.sources_list_box.remove(item_data['box'])
        
        # Remove from tracking list
        self.source_items = [item for item in self.source_items if item['box'] != item_data['box']]
        
        # Update button state
        self.update_next_button_state()

    async def back_to_file_selection(self, widget):
        """Go back to file selection window."""
        self.app.show_file_selection_window()
    
    async def confirm_source_selection(self, widget):
        """Confirm source selection and move to output definition."""
        # Remove the error check since button is only enabled when sources exist
        self.app.show_output_definition_window() 