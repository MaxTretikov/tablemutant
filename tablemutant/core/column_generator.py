# column_generator.py
#!/usr/bin/env python3
"""
ColumnGenerator - Handles DSPy setup and column generation
"""

import json
import logging
from typing import List, Optional
import polars as pl
import dspy
from tqdm import tqdm

logger = logging.getLogger('tablemutant.core.column_generator')


class ColumnGenerator:
    def __init__(self):
        self.lm = None

    def setup_dspy(
        self,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        server_host: str = "http://localhost:8000",
        api_key: str = ""
    ):
        """Configure DSPy to use an OpenAI-compatible API at server_host with optional api_key."""
        logger.debug(
            "setup_dspy temperature: %s, max_tokens: %s, server_host: %s, api_key: %s",
            temperature, max_tokens, server_host, '*' * len(api_key) if api_key else ''
        )

        base = (server_host or "http://localhost:8000").rstrip('/')
        api_base = f"{base}/v1"
        key = api_key if api_key else "dummy"

        # Terminate on our structural end marker
        stop_sequences = ["[[ ## completed ## ]]"]

        self.lm = dspy.LM(
            model='openai/local-model',
            api_base=api_base,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
        )
        dspy.configure(lm=self.lm)
        return self.lm

    def calculate_dynamic_max_tokens(self, source_data_length: int, context_length: int = 0) -> int:
        base_tokens = 200
        input_tokens = (source_data_length + context_length) // 4
        dynamic_output = max(50, min(500, input_tokens * 2))
        total_max_tokens = base_tokens + dynamic_output
        final_max_tokens = min(total_max_tokens, 800)
        logger.debug(
            "Dynamic token calculation: input_len=%d, context_len=%d, input_tokens=%d, "
            "dynamic_output=%d, final_max_tokens=%d",
            source_data_length, context_length, input_tokens, dynamic_output, final_max_tokens
        )
        return final_max_tokens

    def generate_column(
        self,
        df: pl.DataFrame,
        source_columns: List[str],
        instructions: str,
        rag_text: Optional[str] = None,
        example: Optional[str] = None,
        rag_embeddings_data: Optional[List] = None,
        rag_processor=None
    ) -> List[str]:
        """Generate new column values using DSPy with semantic similarity search."""

        class ColumnGeneratorSignature(dspy.Signature):
            """Generate a concise new column value based on source data and instructions."""
            source_data = dspy.InputField(desc="Source column data from the current row")
            task_instructions = dspy.InputField(desc="Instructions for generating the new column")
            context = dspy.InputField(desc="Additional context from RAG source", default="")
            example = dspy.InputField(desc="Example showing input and expected output for reference", default="")
            new_value = dspy.OutputField(desc="The generated value for the new column (be concise)")

        generator = dspy.Predict(ColumnGeneratorSignature)
        new_values = []

        logger.info("Generating new column for %s rows...", len(df))

        use_semantic_search = rag_embeddings_data is not None and rag_processor is not None

        for row in tqdm(df.iter_rows(named=True), total=len(df)):
            # Prepare row payload
            source_data = {}
            for col in source_columns:
                value = row[col]
                if value is None:
                    source_data[col] = "null"
                elif isinstance(value, bytes):
                    try:
                        source_data[col] = value.decode('utf-8', errors='replace')
                    except Exception:
                        source_data[col] = str(value)
                else:
                    source_data[col] = str(value)

            source_str = json.dumps(source_data, indent=2, ensure_ascii=False)

            # Context
            if use_semantic_search:
                all_relevant_chunks = []
                for col in source_columns:
                    v = source_data[col]
                    if v and v.strip() and v != "null":
                        for line in v.split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                            chunks = rag_processor.find_relevant_chunks(line, rag_embeddings_data, top_k=5)
                            if chunks:
                                all_relevant_chunks.extend(chunks)
                seen = set()
                unique_chunks = []
                for c in all_relevant_chunks:
                    if c not in seen:
                        seen.add(c)
                        unique_chunks.append(c)
                context = "\n\n".join(unique_chunks) if unique_chunks else ""
            else:
                context = rag_text or ""

            # Temporary per call max_tokens
            try:
                original_max_tokens = None
                if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'kwargs'):
                    original_max_tokens = dspy.settings.lm.kwargs.get('max_tokens')
                    dspy.settings.lm.kwargs['max_tokens'] = self.calculate_dynamic_max_tokens(
                        len(source_str), len(context)
                    )

                result = generator(
                    source_data=source_str,
                    task_instructions=instructions,
                    context=context,
                    example=example or ""
                )
                new_values.append(getattr(result, "new_value", ""))
            except Exception as e:
                logger.error("Error generating value for row: %s", e)
                new_values.append("")
            finally:
                if original_max_tokens is not None and hasattr(dspy.settings, 'lm'):
                    dspy.settings.lm.kwargs['max_tokens'] = original_max_tokens

        return new_values