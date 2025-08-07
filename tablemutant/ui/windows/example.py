#!/usr/bin/env python3
"""
ExampleWindow - Handles creating examples for output columns
"""

import json
import os
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


class ExampleWindow:
    def __init__(self, app):
        self.app = app
        self.current_definition = None
        self.current_definition_index = None
        self.current_row_number = None
        self.example_input = None
        self.window = None
        
    def show(self, definition, definition_index, row_number=1):
        """Show example window for a specific column definition and row number."""
        self.current_definition = definition
        self.current_definition_index = definition_index
        self.current_row_number = row_number
        
        # Create the example window
        self.window = toga.Window(
            title=f"Add Example for Row {row_number}",
            size=(800, 600),
            resizable=True
        )
        
        self.window.content = self.create_content()
        self.window.show()
    
    def create_content(self):
        """Create and return the example window content."""
        container = toga.Box(style=Pack(direction=COLUMN, padding=20))
        
        # Title
        title = toga.Label(
            "Add Example for Output Column",
            style=Pack(padding=(0, 0, 20, 0), font_size=16, font_weight='bold')
        )
        
        # Get column headers for this definition
        column_headers = [h.value for h in self.current_definition['headers'] if h.value]
        output_column_name = f"'{" - ".join(column_headers)}'" if column_headers else "the output column"
        
        # Instructions
        instructions = toga.Label(
            f"Below are the values from row {self.current_row_number} of your source columns. Enter what {output_column_name} should contain for this example:",
            style=Pack(padding=(0, 0, 15, 0), text_align='left')
        )
        
        # Main horizontal layout
        main_horizontal_section = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 20, 0)))
        
        # Source columns section (left side)
        source_section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 10, 0, 0), flex=1))
        
        # Build column header information
        column_info_parts = []
        if self.app.current_df is not None:
            for col_idx in self.app.selected_columns:
                col_name = self.app.current_df.columns[col_idx]
                column_info_parts.append(col_name)
        
        column_info = " - ".join(column_info_parts) if column_info_parts else "No columns"
        
        source_label = toga.Label(
            f"'{column_info}' value for row {self.current_row_number}:",
            style=Pack(padding=(0, 0, 10, 0), font_weight='bold')
        )
        source_section.add(source_label)
        
        # Get source column data from the specified row
        source_data_text = ""
        if self.app.current_df is not None and len(self.app.current_df) > (self.current_row_number - 1):
            # Get data from the specified row (convert to 0-based index)
            row_data = self.app.current_df.row(self.current_row_number - 1, named=True)
            
            # Format as text for display
            data_lines = []
            for col_idx in self.app.selected_columns:
                col_name = self.app.current_df.columns[col_idx]
                col_value = str(row_data[col_name]) if row_data[col_name] is not None else "null"
                data_lines.append(col_value)
            
            source_data_text = "\n".join(data_lines)
        else:
            source_data_text = f"No source data available for row {self.current_row_number}"
        
        # Source data display (read-only multiline text input)
        source_data_input = toga.MultilineTextInput(
            value=source_data_text,
            readonly=True,
            style=Pack(width=500, height=200, padding=(0, 0, 10, 0))
        )
        
        source_section.add(source_data_input)
        
        # Add document input if documents are present
        if self.app.rag_documents:
            # Get the first document name for the label
            first_doc_name = os.path.basename(self.app.rag_documents[0])
            doc_label_text = f"Retrieval content for {first_doc_name}"
            if len(self.app.rag_documents) > 1:
                doc_label_text += f" (and {len(self.app.rag_documents) - 1} other documents)"
            
            document_label = toga.Label(
                doc_label_text,
                style=Pack(padding=(10, 0, 10, 0), font_weight='bold')
            )
            source_section.add(document_label)
            
            # Get existing document input if any
            existing_document_input = ""
            examples = self.current_definition.get('examples', {})
            if str(self.current_row_number) in examples:
                example_text = examples[str(self.current_row_number)]
                # Try to extract document input from formatted example
                if "Document input: " in example_text:
                    lines = example_text.split('\n')
                    document_lines = []
                    in_document_section = False
                    
                    for line in lines:
                        if line.startswith("Document input: "):
                            # Start of document input section
                            in_document_section = True
                            # Add the content after "Document input: "
                            content = line.replace("Document input: ", "")
                            if content:
                                document_lines.append(content)
                        elif in_document_section and line.startswith("Expected output: "):
                            # End of document input section
                            break
                        elif in_document_section:
                            # Continue collecting document input lines
                            document_lines.append(line)
                    
                    existing_document_input = '\n'.join(document_lines)
            
            self.document_input = toga.MultilineTextInput(
                placeholder="What should I have retrieved?",
                value=existing_document_input,
                style=Pack(width=500, height=100, padding=(0, 0, 10, 0))
            )
            source_section.add(self.document_input)
        else:
            self.document_input = None
        
        # Arrow pointing to output (center)
        arrow_section = toga.Box(style=Pack(direction=COLUMN, padding=(20, 20, 20, 20)))
        arrow_section.add(toga.Box(style=Pack(flex=1)))  # Center the arrow vertically
        arrow_label = toga.Label(
            "→",
            style=Pack(font_size=24, text_align='center')
        )
        arrow_section.add(arrow_label)
        arrow_section.add(toga.Box(style=Pack(flex=1)))  # Center the arrow vertically
        
        # Output section (right side)
        output_section = toga.Box(style=Pack(direction=COLUMN, padding=(0, 0, 0, 10), flex=1))
        output_label = toga.Label(
            f"Expected output for {output_column_name}:",
            style=Pack(padding=(0, 0, 10, 0), font_weight='bold')
        )
        
        # Get existing example if any
        existing_example = ""
        examples = self.current_definition.get('examples', {})
        if str(self.current_row_number) in examples:
            # Extract just the expected output part from the formatted example
            example_text = examples[str(self.current_row_number)]
            if "Expected output: " in example_text:
                existing_example = example_text.split("Expected output: ")[-1]
            else:
                existing_example = example_text
        
        self.example_input = toga.MultilineTextInput(
            placeholder=f"Enter what {output_column_name} should contain for this example...",
            value=existing_example,
            style=Pack(width=500, height=300, padding=(0, 0, 10, 0))
        )
        
        output_section.add(output_label)
        output_section.add(self.example_input)
        
        # Add sections to horizontal layout
        main_horizontal_section.add(source_section)
        main_horizontal_section.add(arrow_section)
        main_horizontal_section.add(output_section)
        
        # Buttons
        button_section = toga.Box(style=Pack(direction=ROW, padding=(20, 0, 0, 0)))
        
        cancel_button = toga.Button(
            "Cancel",
            on_press=self.cancel_example,
            style=Pack(padding=5)
        )
        
        save_button = toga.Button(
            "Save Example",
            on_press=self.save_example,
            style=Pack(padding=5)
        )
        
        button_section.add(cancel_button)
        button_section.add(toga.Box(style=Pack(flex=1)))  # Spacer
        button_section.add(save_button)
        
        # Add all sections to container
        container.add(title)
        container.add(instructions)
        container.add(main_horizontal_section)
        container.add(button_section)
        
        return container
    
    async def save_example(self, widget):
        """Save the example and close window."""
        example_text = self.example_input.value.strip()
        
        if not example_text:
            await self.window.dialog(
                toga.InfoDialog(
                    title="Error",
                    message="Please enter an example output."
                )
            )
            return
        
        # Format the example with source data and expected output
        if self.app.current_df is not None and len(self.app.current_df) > (self.current_row_number - 1):
            # Get data from the specified row (convert to 0-based index)
            row_data = self.app.current_df.row(self.current_row_number - 1, named=True)
            source_data = {}
            
            for col_idx in self.app.selected_columns:
                col_name = self.app.current_df.columns[col_idx]
                col_value = str(row_data[col_name]) if row_data[col_name] is not None else "null"
                source_data[col_name] = col_value
            
            # Format the example
            example_parts = [
                "Example:",
                f"Source data: {json.dumps(source_data, indent=2, ensure_ascii=False)}"
            ]
            
            # Add document input if documents are present and user provided input
            if self.app.rag_documents and self.document_input and self.document_input.value.strip():
                document_input_text = self.document_input.value.strip()
                # Handle multiline document input properly
                if '\n' in document_input_text:
                    # For multiline input, put the first line after "Document input: " and indent subsequent lines
                    lines = document_input_text.split('\n')
                    example_parts.append(f"Document input: {lines[0]}")
                    for line in lines[1:]:
                        example_parts.append(line)
                else:
                    # Single line input
                    example_parts.append(f"Document input: {document_input_text}")
            
            example_parts.append(f"Expected output: {example_text}")
            formatted_example = "\n".join(example_parts)
        else:
            formatted_example = f"Expected output: {example_text}"
        
        # Store the formatted example in the definition's examples dictionary
        if 'examples' not in self.app.column_definitions[self.current_definition_index]:
            self.app.column_definitions[self.current_definition_index]['examples'] = {}
        
        self.app.column_definitions[self.current_definition_index]['examples'][str(self.current_row_number)] = formatted_example
        
        # Update the visual display in the output definition window
        if hasattr(self.app, 'output_window') and self.app.output_window:
            self.app.output_window.update_examples_display(self.current_definition_index)
        
        self.close_window()
    
    async def cancel_example(self, widget):
        """Cancel and close window."""
        self.close_window()
    
    def close_window(self):
        """Close the example window."""
        if self.window:
            self.window.close()
            self.window = None 