#!/usr/bin/env python3
"""
ColumnGenerator - Handles DSPy setup and column generation
"""

import json
from typing import List, Optional
import polars as pl
import dspy
from tqdm import tqdm


class ColumnGenerator:
    def __init__(self):
        self.lm = None
    
    def setup_dspy(self, temperature=0.7, max_tokens=2048, server_host: str = "http://localhost:8000", api_key: str = ""):
        """Configure DSPy to use an OpenAI-compatible API at server_host with optional api_key."""
        import dspy

        base = (server_host or "http://localhost:8000").rstrip('/')
        api_base = f"{base}/v1"
        key = api_key if api_key else "dummy"  # llamafile ignores key; remote may require it

        self.lm = dspy.LM(
            model='openai/local-model',
            api_base=api_base,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        dspy.configure(lm=self.lm)
        
        return self.lm
    
    def generate_column(self, df: pl.DataFrame, source_columns: List[str],
                       instructions: str, rag_text: Optional[str] = None, example: Optional[str] = None,
                       rag_embeddings_data: Optional[List] = None, rag_processor=None) -> List[str]:
        """Generate new column values using DSPy with semantic similarity search."""
        
        class ColumnGeneratorSignature(dspy.Signature):
            """Generate a new column value based on source data and instructions."""
            
            source_data = dspy.InputField(desc="Source column data from the current row")
            task_instructions = dspy.InputField(desc="Instructions for generating the new column")
            context = dspy.InputField(desc="Additional context from RAG source", default="")
            example = dspy.InputField(desc="Example showing input and expected output for reference", default="")
            
            new_value = dspy.OutputField(desc="The generated value for the new column")
        
        # Create a DSPy module
        generator = dspy.ChainOfThought(ColumnGeneratorSignature)
        
        new_values = []
        
        print(f"Generating new column for {len(df)} rows...")
        
        # Check if we should use semantic similarity search
        use_semantic_search = rag_embeddings_data is not None and rag_processor is not None
        
        for row in tqdm(df.iter_rows(named=True), total=len(df)):
            # Prepare source data
            source_data = {}
            for col in source_columns:
                value = row[col]
                # Ensure proper string representation for various types
                if value is None:
                    source_data[col] = "null"
                elif isinstance(value, bytes):
                    # Try to decode bytes
                    try:
                        source_data[col] = value.decode('utf-8', errors='replace')
                    except:
                        source_data[col] = str(value)
                else:
                    source_data[col] = str(value)
            
            source_str = json.dumps(source_data, indent=2, ensure_ascii=False)
            
            # Get context for this row
            if use_semantic_search:
                # Process each non-empty line within each field for semantic similarity search
                all_relevant_chunks = []
                
                for col in source_columns:
                    value = source_data[col]
                    if value and value.strip() and value != "null":
                        # Split the field value by newlines to get individual lines
                        lines = value.split('\n')
                        
                        for line in lines:
                            line = line.strip()
                            if line:  # Only process non-empty lines
                                # Perform semantic similarity search for this line
                                relevant_chunks = rag_processor.find_relevant_chunks(
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
                
                context = "\n\n".join(unique_chunks) if unique_chunks else ""
            else:
                # Use the original RAG text as context
                context = rag_text or ""
            
            # Generate new value
            try:
                result = generator(
                    source_data=source_str,
                    task_instructions=instructions,
                    context=context,
                    example=example or ""
                )
                new_values.append(getattr(result, "new_value", ""))
            except Exception as e:
                print(f"Error generating value for row: {e}")
                new_values.append("")  # Default empty value on error
        
        return new_values