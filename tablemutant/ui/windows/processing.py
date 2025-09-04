# processing.py
#!/usr/bin/env python3
"""
ProcessingWindow - Handles showing progress during generation processing
"""

import asyncio
import json
import os
import time
import logging
import re
import threading
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import toga
from toga.style import Pack
from toga.style.pack import COLUMN
import dspy
import polars as pl

from tablemutant.core.tls_config import get_http_client

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
        self.live_output_header = None
        self.live_output_display = None
        self.cancel_button = None
        self.processing_cancelled = False

        # Dedicated thread pools prevent starvation in packaged apps
        cpu = os.cpu_count() or 4
        self.io_executor = ThreadPoolExecutor(
            max_workers=max(4, cpu),
            thread_name_prefix="tm-io"
        )
        # Model work is now handled by a single long‑lived thread with streaming updates
        self.model_thread = None

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.generation_start_time = None
        self.token_history = deque(maxlen=10)  # recent completion token speeds

        # Per‑definition first‑row metrics used for ETA
        self._eta_ratio_by_def = {}          # def_key -> output_tokens/input_tokens
        self._eta_tps_by_def = {}            # def_key -> tokens/sec measured from first row

        # Completion synchronization with the worker thread
        self._ui_loop = None
        self._worker_done_event = None
        self._worker_result = None

    def create_content(self, preview_only=True):
        self.process_box = toga.Box(style=Pack(direction=COLUMN, margin=10))
        title = toga.Label(
            "Processing..." if not preview_only else "Generating Preview...",
            style=Pack(margin=(0, 0, 20, 0), font_size=16, font_weight='bold')
        )
        self.process_status = toga.Label("Initializing model...", style=Pack(margin=(0, 0, 10, 0)))
        self.process_progress = toga.ProgressBar(max=100, style=Pack(width=500, margin=(0, 0, 10, 0)))
        self.time_estimate = toga.Label("Estimating time...", style=Pack(margin=(0, 0, 10, 0)))
        self.tokens_info = toga.Label("Tokens: 0 | Speed: -- tokens/sec", style=Pack(margin=(0, 0, 20, 0)))
        self.prompt_header = toga.Label("Current Prompt:", style=Pack(margin=(0, 0, 5, 0), font_size=14, font_weight='bold'))
        self.prompt_display = toga.MultilineTextInput(readonly=True, placeholder="Current prompt will appear here...", style=Pack(width=500, height=200, margin=(0, 0, 20, 0)))
        # New: live streamed output view placed under the prompt
        self.live_output_header = toga.Label("Live Output:", style=Pack(margin=(0, 0, 5, 0), font_size=14, font_weight='bold'))
        self.live_output_display = toga.MultilineTextInput(readonly=True, placeholder="Model tokens will stream here", style=Pack(width=500, height=220, margin=(0, 0, 20, 0)))
        self.cancel_button = toga.Button("Cancel", on_press=self.cancel_processing, style=Pack(margin=5))

        self.process_box.add(title)
        self.process_box.add(self.process_status)
        self.process_box.add(self.process_progress)
        self.process_box.add(self.time_estimate)
        self.process_box.add(self.tokens_info)
        self.process_box.add(self.prompt_header)
        self.process_box.add(self.prompt_display)
        self.process_box.add(self.live_output_header)
        self.process_box.add(self.live_output_display)
        self.process_box.add(self.cancel_button)
        return self.process_box

    async def update_progress(self, value, status_text=None, time_text=None, tokens_text=None):
        self.process_progress.value = value
        if status_text:
            self.process_status.text = status_text
        if time_text:
            self.time_estimate.text = time_text
        if tokens_text:
            self.tokens_info.text = tokens_text
        await asyncio.sleep(0.01)

    async def update_prompt_display(self, prompt_text, header_text=None):
        if self.prompt_display:
            self.prompt_display.value = prompt_text
        if header_text and self.prompt_header:
            self.prompt_header.text = header_text
        await asyncio.sleep(0.01)

    async def reset_live_output(self, title_suffix=""):
        if self.live_output_header:
            self.live_output_header.text = f"Live Output{title_suffix}"
        if self.live_output_display:
            self.live_output_display.value = ""
        await asyncio.sleep(0.01)

    async def append_live_output(self, text):
        if self.live_output_display:
            self.live_output_display.value = (self.live_output_display.value or "") + text
        await asyncio.sleep(0.0)

    def _post_ui(self, coro):
        """Schedule a coroutine onto the UI loop from any thread."""
        if self._ui_loop is not None:
            asyncio.run_coroutine_threadsafe(coro, self._ui_loop)

    def format_dspy_prompt(self, signature_class, inputs):
        lines = []
        if getattr(signature_class, '__doc__', None):
            lines.append(f"Task: {signature_class.__doc__.strip()}")
            lines.append("")
        for name, field in signature_class.__annotations__.items():
            desc = getattr(field, 'desc', name)
            if name in inputs:
                lines.append(f"{desc}:")
                lines.append(str(inputs[name]))
                lines.append("")
        outs = []
        for name, field in signature_class.__annotations__.items():
            if name not in inputs:
                outs.append(f"- {name}: {getattr(field, 'desc', name)}")
        if outs:
            lines.append("Expected Output:")
            lines.extend(outs)
            lines.append("")
        lines.append("Please generate the requested output based on the input data above.")
        return "\n".join(lines)

    async def process_generation(self, definitions=None, preview_only=True):
        try:
            await self.update_progress(10, "Setting up model...")
            loop = asyncio.get_running_loop()
            self._ui_loop = loop

            dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False, enable_litellm_cache=False)

            # Liveness probe off the UI loop
            server_host = self.app.settings_manager.get('server_host')
            auth_token = self.app.settings_manager.get('auth_token')
            models_url = (server_host.rstrip('/')) + '/v1/models'

            def _is_local(url: str) -> bool:
                try:
                    from urllib.parse import urlparse
                    host = (urlparse(url).hostname or '').lower()
                    return host in ('localhost', '127.0.0.1', '::1')
                except Exception:
                    return True

            headers = {}
            if not _is_local(server_host) and auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'

            def _probe(url, hdrs):
                try:
                    with get_http_client().urlopen(url, timeout=2, headers=hdrs) as resp:
                        return resp.status
                except Exception:
                    return None

            status = await loop.run_in_executor(self.io_executor, _probe, models_url, headers)
            if status == 200:
                await self.update_progress(40, "Server already running!" if not _is_local(server_host) else "Llamafile server already running!")
                self.app.server_was_running = True
            else:
                self.app.server_was_running = False

            if self.app.server_was_running:
                await self.update_progress(45, "Configuring DSPy...")
                try:
                    self.app.tm.setup_dspy_configuration()
                    await self.update_progress(50, "DSPy configuration complete!")
                except Exception as e:
                    await self.update_progress(46, "Retrying DSPy configuration...")
                    try:
                        from tablemutant.core.column_generator import ColumnGenerator
                        self.app.tm.column_generator = ColumnGenerator()
                        self.app.tm.setup_dspy_configuration()
                        await self.update_progress(50, "DSPy configuration complete!")
                    except Exception as e2:
                        raise Exception(f"Failed to configure DSPy: {e2}")

            if not self.app.server_was_running:
                if _is_local(server_host):
                    await self.update_progress(20, "Setting up model and local server...")

                    def setup_model_and_server():
                        return self.app.tm.setup_model_and_server_only()

                    try:
                        await loop.run_in_executor(self.io_executor, setup_model_and_server)
                        await self.update_progress(40, "Model and server setup complete!")
                    except Exception as e:
                        raise Exception(f"Failed to setup model: {e}")
                else:
                    await self.update_progress(40, "Using remote server endpoint...")

                await self.update_progress(45, "Configuring DSPy...")
                try:
                    self.app.tm.setup_dspy_configuration()
                    await self.update_progress(50, "DSPy configuration complete!")
                except Exception:
                    await self.update_progress(46, "Retrying DSPy configuration...")
                    from tablemutant.core.column_generator import ColumnGenerator
                    self.app.tm.column_generator = ColumnGenerator()
                    self.app.tm.setup_dspy_configuration()
                    await self.update_progress(50, "DSPy configuration complete!")

            if self.processing_cancelled:
                return

            # Filter rows off the UI loop
            if preview_only:
                df_to_process = await loop.run_in_executor(self.io_executor, self.get_non_empty_rows, self.app.current_df, self.app.preview_rows)
            else:
                df_to_process = await loop.run_in_executor(self.io_executor, self.get_non_empty_rows, self.app.current_df, None)
            num_rows = len(df_to_process)
            source_column_names = [self.app.current_df.columns[i] for i in self.app.selected_columns]

            # Load RAG sources off the UI loop
            rag_embeddings_data = []
            if self.app.rag_documents:
                await self.update_progress(55, "Loading RAG documents...")
                for doc_path in self.app.rag_documents:
                    try:
                        cached = await loop.run_in_executor(self.io_executor, self.app.tm.rag_processor.get_cached_embeddings, doc_path)
                        if cached:
                            text_chunks, embeddings, metadata = cached
                            rag_embeddings_data.append({'path': doc_path, 'chunks': text_chunks, 'embeddings': embeddings, 'metadata': metadata})
                        else:
                            doc_text = await loop.run_in_executor(self.io_executor, self.app.tm.rag_processor.load_rag_source, doc_path)
                            if doc_text:
                                chunks = [c.strip() for c in doc_text.split('\n\n') if c.strip()]
                                rag_embeddings_data.append({'path': doc_path, 'chunks': chunks, 'embeddings': None, 'metadata': {'fallback': True}})
                    except Exception as e:
                        logger.error("Error loading RAG document %s: %s", doc_path, e)

            # Kick off the streaming generation in a dedicated worker thread
            total_operations = len(definitions) * max(0, num_rows)
            self.generation_start_time = time.time()
            self._worker_done_event = asyncio.Event()

            def _done_callback(result_columns):
                self._worker_result = result_columns
                if self._ui_loop:
                    self._ui_loop.call_soon_threadsafe(self._worker_done_event.set)

            # Arguments packed for the worker thread
            args = dict(
                server_host=server_host,
                headers=headers,
                df_to_process=df_to_process,
                source_column_names=source_column_names,
                rag_embeddings_data=rag_embeddings_data,
                definitions=definitions,
                total_operations=total_operations,
                preview_only=preview_only,
                done_callback=_done_callback
            )
            self.model_thread = threading.Thread(target=self._generation_worker_thread, kwargs=args, daemon=True)
            self.model_thread.start()

            # Wait for completion without blocking the UI loop
            await self._worker_done_event.wait()

            if self.processing_cancelled:
                return

            await self.update_progress(100, "Generation complete!")
            await asyncio.sleep(0.25)

            if preview_only:
                self.app.show_preview_results_window(df_to_process, self._worker_result or {}, definitions)
            else:
                await self.save_results(self._worker_result or {})

        except Exception as e:
            await self.app.main_window.dialog(toga.ErrorDialog(title="Error", message=f"Processing failed: {str(e)}"))
            self.app.show_output_definition_window()
        finally:
            if not self.app.server_was_running:
                self.app.tm.cleanup()
            self.io_executor.shutdown(wait=False)
            # model_thread is daemon; no explicit join on purpose

    def get_non_empty_rows(self, df, max_rows=None):
        if df is None or df.is_empty() or not self.app.selected_columns:
            return df.head(0) if df is not None else None
        column_names = [df.columns[i] for i in self.app.selected_columns if i < len(df.columns)]
        if not column_names:
            return df.head(0)
        non_empty_rows = []
        for i in range(len(df)):
            has_val = False
            for col_name in column_names:
                try:
                    v = df[col_name][i]
                    if v is not None:
                        s = str(v).strip()
                        if s and s.lower() not in ['null', 'na', 'n/a', '']:
                            has_val = True
                            break
                except Exception:
                    continue
            if has_val:
                non_empty_rows.append(i)
                if max_rows is not None and len(non_empty_rows) >= max_rows:
                    break
        return df[non_empty_rows] if non_empty_rows else df.head(0)

    def calculate_tokens_text(self):
        if self.generation_start_time and self.total_tokens > 0:
            if self.token_history:
                recent = sum(self.token_history) / len(self.token_history)
            else:
                recent = 0
            return (f"Tokens: {self.total_tokens:,} total "
                    f"({self.total_prompt_tokens:,} prompt, {self.total_completion_tokens:,} completion) | "
                    f"Speed: {recent:.1f} tokens/sec")
        return "Tokens: 0 | Speed: -- tokens/sec"

    async def cancel_processing(self, widget):
        self.processing_cancelled = True
        await self.update_progress(0, "Cancelling...")
        self.app.tm.cleanup()
        await asyncio.sleep(1)
        self.app.show_output_definition_window()

    async def save_results(self, new_columns):
        all_non_empty_df = self.get_non_empty_rows(self.app.current_df, None)
        preview_rows_count = len(new_columns[list(new_columns.keys())[0]]) if new_columns else 0
        if preview_rows_count > 0 and preview_rows_count < len(all_non_empty_df):
            remaining_rows_df = all_non_empty_df[preview_rows_count:]
            for col_name, preview_values in new_columns.items():
                remaining_count = len(remaining_rows_df)
                new_columns[col_name] = preview_values + [""] * remaining_count

        result_df = self.app.original_df.clone()

        for col_name, values in new_columns.items():
            full_values = []
            for _ in range(self.app.header_rows):
                full_values.append("")
            non_empty_indices = []
            for i in range(len(self.app.current_df)):
                has_non_empty = False
                for col_idx in self.app.selected_columns:
                    if col_idx < len(self.app.current_df.columns):
                        col_name_check = self.app.current_df.columns[col_idx]
                        try:
                            value = self.app.current_df[col_name_check][i]
                            if value is not None:
                                s = str(value).strip()
                                if s and s.lower() not in ['null', 'na', 'n/a', '']:
                                    has_non_empty = True
                                    break
                        except Exception:
                            continue
                if has_non_empty:
                    non_empty_indices.append(i)
            working_full_values = [""] * len(self.app.current_df)
            for idx, value in enumerate(values):
                if idx < len(non_empty_indices):
                    working_full_values[non_empty_indices[idx]] = value
            full_values.extend(working_full_values)
            if len(full_values) < len(result_df):
                full_values.extend([""] * (len(result_df) - len(full_values)))
            elif len(full_values) > len(result_df):
                full_values = full_values[:len(result_df)]
            new_col_name = f"generated_{len([k for k in new_columns.keys() if k <= col_name])}" if col_name in result_df.columns else col_name
            result_df = result_df.with_columns(pl.Series(name=new_col_name, values=full_values))

        output_path = self.app.table_path.rsplit('.', 1)[0] + '_mutated.' + self.app.table_path.rsplit('.', 1)[1]
        self.app.tm.table_processor.save_table(result_df, output_path)
        await self.app.main_window.dialog(toga.InfoDialog(title="Success", message=f"Results saved to: {output_path}\n\nPreview generations were used at the start of the file."))
        self.app.main_window.close()

    # -------------------------
    # Worker thread and helpers
    # -------------------------

    def _approx_tokens(self, text_len: int) -> int:
        # Heuristic compatible with llama.cpp tokens; roughly 4 chars per token for English
        return max(1, text_len // 4)

    def _build_prompt_text(self, row_data, defn, display_rag_context):
        class _Sig(dspy.Signature):
            """Generate a new column value based on source data and instructions."""
            source_data = dspy.InputField(desc="Source column data from the current row")
            task_instructions = dspy.InputField(desc="Instructions for generating the new column")
            context = dspy.InputField(desc="Additional context from RAG documents")
            example = dspy.InputField(desc="Example showing input and expected output for reference", default="")
            new_value = dspy.OutputField(desc="The generated value for the new column")
        base = self.format_dspy_prompt(
            _Sig,
            {
                "source_data": json.dumps(row_data, indent=2, ensure_ascii=False),
                "task_instructions": defn['instructions'],
                "context": display_rag_context,
                "example": defn.get('example', '') or ""
            }
        )
        # Enforce strict structured output for downstream parsing
        contract = (
            "\nOutput format:\n"
            '- Return a single JSON object with exactly one key "new_value" whose value is the final answer string.\n'
            "- Do not include code fences; do not add extra keys; do not prefix with list markers.\n"
            "- End your reply with the exact marker [[ ## completed ## ]].\n"
        )
        return base + "\n" + contract

    def _gather_relevant_chunks(self, row_data, rag_embeddings_data, top_k=5):
        if not rag_embeddings_data:
            return []
        all_chunks = []
        for _, val in row_data.items():
            if not val:
                continue
            s = str(val).strip()
            if not s or s == "null":
                continue
            for line in s.split('\n'):
                line = line.strip()
                if not line:
                    continue
                rel = self.app.tm.rag_processor.find_relevant_chunks(line, rag_embeddings_data, top_k=top_k)
                if rel:
                    all_chunks.extend(rel)
        seen = set()
        uniq = []
        for c in all_chunks:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def _stream_chat_completion(self, server_host, headers, model, prompt_text, stop, max_tokens):
        """
        Synchronous streaming call to the OpenAI‑compatible /v1/chat/completions endpoint.
        Yields deltas to the UI and returns (full_text, prompt_tokens_est, completion_tokens_est, elapsed_sec).
        """
        url = (server_host.rstrip('/')) + '/v1/chat/completions'
        payload = {
            "model": model or "openai/local-model",
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": getattr(getattr(dspy, "settings", None), "temperature", None) or 0.7,
            "max_tokens": max_tokens,
            "stop": stop or [],
            "stream": True
        }
        req_headers = dict(headers or {})
        req_headers["Content-Type"] = "application/json"
        req_headers["Accept"] = "text/event-stream"

        client = get_http_client()
        start = time.time()
        full = []
        usage_prompt = self._approx_tokens(len(prompt_text))
        usage_completion = 0

        # Stream response using the correct HTTPClient API
        try:
            # Use the request method with POST and data parameter
            resp = client.request(
                url, method="POST", timeout=30, headers=req_headers, data=json.dumps(payload).encode("utf-8")
            )
            
            # Read the streaming response in chunks
            buf = b""
            while True:
                if self.processing_cancelled:
                    try:
                        resp.close()
                    except Exception:
                        pass
                    break
                
                try:
                    chunk = resp.read(2048)
                    if not chunk:
                        break
                except Exception:
                    break
                    
                buf += chunk
                while b"\n\n" in buf:
                    frame, buf = buf.split(b"\n\n", 1)
                    for line in frame.split(b"\n"):
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            break
                        try:
                            obj = json.loads(data.decode("utf-8"))
                        except Exception:
                            continue
                        choices = obj.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta", {})
                            txt = delta.get("content")
                            if txt:
                                full.append(txt)
                                usage_completion += self._approx_tokens(len(txt))
                                # push token to UI
                                self._post_ui(self.append_live_output(txt))
                        # If server includes usage in the final streamed object; honor it
                        usage = obj.get("usage")
                        if usage:
                            usage_prompt = usage.get("prompt_tokens", usage_prompt)
                            usage_completion = usage.get("completion_tokens", usage_completion)
            try:
                resp.close()
            except Exception:
                pass
        except Exception as e:
            logger.error("Streaming error: %s", e)
            raise

        elapsed = max(time.time() - start, 1e-6)
        return "".join(full), usage_prompt, usage_completion, elapsed

    # -------------------------
    # Structured output parsing
    # -------------------------
    def _strip_markers_and_fences(self, text: str) -> str:
        if not text:
            return ""
        s = text
        # Remove known stop markers and chatter
        for marker in ("[[ ## completed ## ]]", "<end_of_turn>"):
            s = s.replace(marker, "")
        # Strip common code fences
        fence_re = re.compile(r"```(?:json|yaml|yml|text)?\s*([\s\S]*?)```", re.IGNORECASE)
        while True:
            m = fence_re.search(s)
            if not m:
                break
            s = s[:m.start()] + m.group(1) + s[m.end():]
        return s.strip()

    def _iter_json_objects(self, s: str):
        """Yield balanced JSON object substrings from s."""
        stack = 0
        start = None
        in_str = False
        esc = False
        for i, ch in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if stack == 0:
                    start = i
                stack += 1
            elif ch == "}":
                if stack:
                    stack -= 1
                    if stack == 0 and start is not None:
                        yield s[start:i+1]
                        start = None

    def _try_json_extract_new_value(self, s: str):
        for obj_str in self._iter_json_objects(s):
            try:
                obj = json.loads(obj_str)
            except Exception:
                continue
            if isinstance(obj, dict) and "new_value" in obj:
                v = obj.get("new_value")
                if v is None:
                    return ""
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return None

    def _unquote_scalar(self, v: str) -> str:
        t = v.strip()
        if not t:
            return ""
        if t.startswith('"') and t.endswith('"'):
            try:
                return json.loads(t)
            except Exception:
                return t[1:-1]
        if t.startswith("'") and t.endswith("'"):
            # Convert to JSON‑compatible double quotes naïvely
            j = '"' + t[1:-1].replace('"', '\\"') + '"'
            try:
                return json.loads(j)
            except Exception:
                return t[1:-1]
        return t

    def _parse_yamlish_new_value(self, s: str):
        # Collect all lines like "- new_value: ...", or "new_value: ..."
        lines = s.splitlines()
        hits = []
        pat = re.compile(r"^\s*(?:-\s*)?new_value\s*:\s*(.*)$", re.IGNORECASE)
        for ln in lines:
            m = pat.match(ln)
            if m:
                val = self._unquote_scalar(m.group(1))
                # Unescape visible \n sequences
                val = val.replace("\\n", "\n")
                hits.append(val)
        if hits:
            # Multiple entries collapse to a newline‑joined single new_value
            return "\n".join(hits)
        return None

    def _extract_new_value(self, raw_text: str) -> str:
        s = self._strip_markers_and_fences(raw_text or "")
        # 1) Strict JSON object with "new_value"
        v = self._try_json_extract_new_value(s)
        if v is not None:
            return v.strip()
        # 2) YAML‑ish fallbacks
        v = self._parse_yamlish_new_value(s)
        if v is not None:
            return v.strip()
        # 3) Last resort; return the body as is
        return s.strip()

    def _generation_worker_thread(
        self,
        *,
        server_host,
        headers,
        df_to_process,
        source_column_names,
        rag_embeddings_data,
        definitions,
        total_operations,
        preview_only,
        done_callback
    ):
        """
        Runs in a single background thread. Streams tokens; posts granular UI updates.
        """
        try:
            all_new_columns = {}
            completed = 0
            total_ops = max(1, total_operations)
            model_name = getattr(getattr(dspy, "settings", None), "lm", None)
            model_name = getattr(model_name, "model", None) or "openai/local-model"
            stop_sequences = ["[[ ## completed ## ]]"]

            for def_index, defn in enumerate(definitions):
                if self.processing_cancelled:
                    break
                column_name = " - ".join(defn['headers'])
                new_values = []
                def_key = column_name
                self._post_ui(self.update_progress(
                    50 + (completed / total_ops) * 50,
                    f"Generating column: {column_name}", None, self.calculate_tokens_text()
                ))

                # Precompute input token estimates for this definition for ETA
                row_input_est = []
                for row_idx in range(len(df_to_process)):
                    # Build light row payload only for token estimation
                    est_row = {}
                    for col in source_column_names:
                        try:
                            v = df_to_process[col][row_idx]
                            if v is None:
                                est_row[col] = "null"
                            elif isinstance(v, bytes):
                                try:
                                    est_row[col] = v.decode('utf-8', errors='replace')
                                except Exception:
                                    est_row[col] = str(v)
                            else:
                                est_row[col] = str(v)
                        except Exception:
                            est_row[col] = ""
                    est_len = len(json.dumps(est_row, ensure_ascii=False))
                    row_input_est.append(self._approx_tokens(est_len))

                for row_idx in range(len(df_to_process)):
                    if self.processing_cancelled:
                        break

                    # Full row payload for generation
                    row_data = {}
                    for col in source_column_names:
                        try:
                            v = df_to_process[col][row_idx]
                            if v is None:
                                row_data[col] = "null"
                            elif isinstance(v, bytes):
                                try:
                                    row_data[col] = v.decode('utf-8', errors='replace')
                                except Exception:
                                    row_data[col] = str(v)
                            else:
                                row_data[col] = str(v)
                        except Exception:
                            row_data[col] = ""

                    # RAG display context for prompt view
                    display_chunks = self._gather_relevant_chunks(row_data, rag_embeddings_data, top_k=5)
                    display_rag_context = "\n\nRelevant context from documents:\n" + "\n".join(display_chunks) if display_chunks else ""

                    prompt_text = self._build_prompt_text(row_data, defn, display_rag_context)
                    self._post_ui(self.update_prompt_display(prompt_text, f"Generation Prompt (Row {row_idx + 1})"))
                    self._post_ui(self.reset_live_output(f" (Row {row_idx + 1})"))

                    # Dynamic output limit based on prompt and RAG size
                    dyn_max = self.app.tm.column_generator.calculate_dynamic_max_tokens(
                        len(json.dumps(row_data, ensure_ascii=False)), len(display_rag_context)
                    )

                    # Stream completion
                    try:
                        text, pt, ct, dt = self._stream_chat_completion(
                            server_host, headers, model_name, prompt_text, stop_sequences, dyn_max
                        )
                    except Exception as e:
                        text = f"Error: {str(e)[:200]}"
                        pt = self._approx_tokens(len(prompt_text))
                        ct = self._approx_tokens(len(text))
                        dt = 0.01

                    # Track usage and speed
                    self.total_prompt_tokens += pt
                    self.total_completion_tokens += ct
                    self.total_tokens = self.total_prompt_tokens + self.total_completion_tokens
                    if ct > 0 and dt > 0:
                        self.token_history.append(ct / dt)

                    # Parse DSPy‑style structured output; keep only the new_value string
                    parsed_value = self._extract_new_value(text)
                    new_values.append(parsed_value)
                    completed += 1

                    # After the first completed row in a definition; derive ETA parameters
                    if def_key not in self._eta_ratio_by_def:
                        ipt_est = max(1, row_input_est[row_idx])
                        ratio = ct / float(ipt_est)
                        tps = max(1e-6, ct / dt)
                        self._eta_ratio_by_def[def_key] = max(0.2, min(8.0, ratio))  # clamp to sane range
                        self._eta_tps_by_def[def_key] = tps

                    # Compute remaining ETA using first‑row metrics
                    remain_rows = len(df_to_process) - (row_idx + 1)
                    if remain_rows > 0:
                        ratio = self._eta_ratio_by_def.get(def_key, 1.0)
                        tps = self._eta_tps_by_def.get(def_key, sum(self.token_history) / max(1, len(self.token_history)) or 1.0)
                        # Sum predicted completion tokens for remaining rows in current definition
                        pred_ct = 0
                        base_idx = row_idx + 1
                        for k in range(base_idx, len(row_input_est)):
                            pred_ct += ratio * max(1, row_input_est[k])
                        eta_sec = int(pred_ct / max(1e-6, tps))
                        time_text = f"Remaining: ~{timedelta(seconds=eta_sec)}"
                    else:
                        time_text = "Remaining: ~0:00"

                    progress = 50 + (completed / total_ops) * 50
                    self._post_ui(self.update_progress(progress, f"Generating column: {column_name} (row {row_idx + 1}/{len(df_to_process)})", time_text, self.calculate_tokens_text()))

                all_new_columns[column_name] = new_values

            done_callback(all_new_columns)
        except Exception as e:
            logger.error("Worker failure: %s", e)
            done_callback({})