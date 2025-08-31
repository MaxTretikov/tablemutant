#!/usr/bin/env python3
"""
ModelManager - Handles model downloading and llamafile operations
"""

import subprocess
import sys
import os
import platform
import signal
import atexit
import time
import threading
import urllib.parse
import json
import logging
import tempfile
import contextlib
import hashlib
import errno
import re
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from collections import deque

from .tls_config import get_http_client

logger = logging.getLogger('tablemutant.core.model_manager')


class ModelManager:
    def __init__(self):
        self.llamafile_process = None
        self.llamafile_path = None
        self._dl_lock = threading.Lock()
        self.model_path = None
        self._log_tail = deque(maxlen=400)
        self._log_thread = None

        # Import here to avoid circular imports
        from .settings_manager import SettingsManager
        self._settings_manager = SettingsManager()

        # Use centralized HTTP client
        self._http_client = get_http_client()
    
    # -----------------------------
    # Download helpers
    # -----------------------------

    @contextlib.contextmanager
    def _file_lock(self, lock_path: Path, poll=0.05):
        """Cross-platform advisory lock via exclusive file create. Single writer only."""
        lock_path = Path(lock_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        fd = None
        while True:
            try:
                # O_EXCL + O_CREAT succeeds only if the file does not exist
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except OSError as e:
                if e.errno == errno.EEXIST:
                    time.sleep(poll)
                    continue
                raise

        try:
            yield
        finally:
            try:
                if fd is not None:
                    os.close(fd)
            finally:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
    
    def _validate_llamafile_binary(self, path: Path) -> bool:
        try:
            st = path.stat()
            # sanity: llamafile is huge; avoid HTML error pages etc.
            if st.st_size < 50 * 1024 * 1024:
                return False

            with open(path, "rb") as f:
                head = f.read(4096)

            if not (head.startswith(b"#!") or self._is_probably_binary(head)):
                return False

            # Best-effort smoke test; do not fail the download if it times out
            try:
                r = subprocess.run(
                    [str(path), "--version"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=8,
                    start_new_session=True,
                )
                out = (r.stdout or "").lower()
                # accept common outputs; do not overfit to one string
                if ("llamafile" in out) or ("llama.cpp" in out) or ("server" in out) or (r.returncode == 0 and out):
                    return True
            except Exception:
                # Execution test is flaky across macOS versions; magic+size already passed
                return True

            return True  # size+magic passed; treat as valid
        except Exception:
            return False


    # -----------------------------
    # Paths and downloads
    # -----------------------------

    def get_models_dir(self) -> Path:
        """Resolve a single, stable models directory consistent with SettingsManager."""
        home_override = os.environ.get("TABLEMUTANT_HOME")
        models_override = os.environ.get("TABLEMUTANT_MODELS_DIR")
        logger.debug("get_models_dir home_override: %s, models_override: %s", home_override, models_override)

        if models_override:
            models_dir = Path(models_override)
        elif home_override:
            models_dir = Path(home_override) / "models"
        else:
            models_dir = self._settings_manager.get_app_base_dir() / "models"

        logger.debug("get_models_dir returning: %s", models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    def validate_gguf(self, model_identifier: str) -> bool:
        """Validate that the model identifier points to a GGUF file."""
        logger.debug("validate_gguf model_identifier: %s", model_identifier)
        if os.path.exists(model_identifier):
            result = model_identifier.endswith(".gguf")
            logger.debug("validate_gguf file path result: %s", result)
            return result

        parsed = urllib.parse.urlparse(model_identifier)
        if parsed.scheme in ["http", "https"]:
            result = parsed.path.endswith(".gguf")
            logger.debug("validate_gguf URL result: %s", result)
            return result

        parts = model_identifier.split("/")
        if len(parts) >= 2:
            if len(parts) > 2 and parts[-1].endswith(".gguf"):
                logger.debug("validate_gguf HuggingFace with filename: True")
                return True
            result = "gguf" in model_identifier.lower()
            logger.debug("validate_gguf HuggingFace result: %s", result)
            return result

        logger.debug("validate_gguf default result: False")
        return False

    def download_model(self, model_identifier: str) -> str:
        """Download model if it is not local. Returns path to local file."""
        logger.debug("download_model model_identifier: %s", model_identifier)
        if os.path.exists(model_identifier):
            result = os.path.abspath(model_identifier)
            logger.debug("download_model returning existing file: %s", result)
            return result

        models_dir = self.get_models_dir()

        if "/" in model_identifier and not model_identifier.startswith("http"):
            try:
                from huggingface_hub import hf_hub_download, list_repo_files

                parts = model_identifier.split("/")
                repo_id = "/".join(parts[:2])
                local_repo_dir = models_dir / repo_id.replace("/", "_")
                local_repo_dir.mkdir(parents=True, exist_ok=True)

                def _prefer_local_gguf(path: Path) -> Optional[Path]:
                    ggufs = sorted(path.glob("*.gguf"))
                    if not ggufs:
                        return None
                    quantized = [p for p in ggufs if "Q" in p.name.upper()]
                    if quantized:
                        quantized_sorted = sorted(quantized, key=lambda p: p.name.upper())
                        return quantized_sorted[0]
                    return ggufs[0]

                if len(parts) > 2:
                    filename = parts[-1]
                    candidate = local_repo_dir / filename
                    if candidate.exists():
                        return str(candidate.resolve())
                else:
                    local_existing = _prefer_local_gguf(local_repo_dir)
                    if local_existing:
                        return str(local_existing.resolve())

                    files = list_repo_files(repo_id)
                    gguf_files = [f for f in files if f.endswith(".gguf")]
                    if not gguf_files:
                        raise ValueError(f"No GGUF files found in repository {repo_id}")

                    quantized = [f for f in gguf_files if "Q" in f.upper()]
                    if quantized:
                        gguf_files = sorted(quantized, key=lambda n: n.upper())
                    filename = gguf_files[0]

                logger.info("Downloading %s from %s...", filename, repo_id)
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=models_dir,
                    local_dir=local_repo_dir,
                )
                return str(Path(local_path).resolve())

            except ImportError:
                logger.error("huggingface-hub not installed. Install with: pip install huggingface-hub")
                sys.exit(1)
            except Exception as e:
                logger.error("Error downloading from HuggingFace: %s", e)
                sys.exit(1)

        elif model_identifier.startswith("http"):
            filename = os.path.basename(urllib.parse.urlparse(model_identifier).path)
            local_path = models_dir / filename

            if local_path.exists():
                logger.info("Model already downloaded: %s", local_path)
                return str(local_path)

            logger.info("Downloading %s...", model_identifier)

            with self._http_client.urlopen(model_identifier) as response:
                total_size = int(response.headers.get("content-length", 0))
                with open(local_path, "wb") as f:
                    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))

            return str(local_path)

        raise ValueError(f"Unknown model identifier format: {model_identifier}")

    def get_bin_dir(self) -> Path:
        """Get the appropriate bin directory based on the platform."""
        home_override = os.environ.get("TABLEMUTANT_HOME")
        bin_override = os.environ.get("TABLEMUTANT_BIN_DIR")
        logger.debug("get_bin_dir home_override: %s, bin_override: %s", home_override, bin_override)

        if bin_override:
            bin_dir = Path(bin_override)
        elif home_override:
            bin_dir = Path(home_override) / "bin"
        else:
            bin_dir = self._settings_manager.get_app_base_dir() / "bin"

        logger.debug("get_bin_dir returning: %s", bin_dir)
        bin_dir.mkdir(parents=True, exist_ok=True)
        return bin_dir

    def download_llamafile(self, progress_cb=None) -> str:
        """Download the latest llamafile if not present; serialize work; atomic write; optional progress callback."""
        llamafile_dir = self.get_bin_dir()
        logger.debug("download_llamafile llamafile_dir: %s", llamafile_dir)

        lockfile = llamafile_dir / ".llamafile.lock"

        with self._dl_lock, self._file_lock(lockfile):
            # Recheck after acquiring locks
            existing = sorted(llamafile_dir.glob("llamafile-*"), key=os.path.getctime)
            if existing:
                candidate = existing[-1]
                if self._validate_llamafile_binary(candidate):
                    logger.info("Using existing llamafile: %s", candidate)
                    return str(candidate.resolve())
                else:
                    logger.warning("Existing llamafile candidate is invalid; will redownload: %s", candidate)

            logger.info("Fetching latest llamafile release...")
            with self._http_client.urlopen(
                "https://api.github.com/repos/Mozilla-Ocho/llamafile/releases/latest",
                headers={"Accept": "application/vnd.github+json"},
            ) as response:
                release_data = json.loads(response.read().decode())

            assets = release_data.get("assets", [])
            candidates = []
            for a in assets:
                name = a.get("name", "")
                if not name.startswith("llamafile-"):
                    continue
                if any(x in name for x in [".zip", ".tar", ".exe", ".gz", ".sig", ".sha", "bench"]):
                    continue
                size = int(a.get("size", 0))
                if size < 50 * 1024 * 1024:
                    continue
                candidates.append(a)

            if not candidates:
                raise ValueError("Could not find a suitable llamafile asset in the latest release")
            asset = max(candidates, key=lambda a: int(a.get("size", 0)))

            target = llamafile_dir / asset["name"]
            tmp = target.with_suffix(target.suffix + ".partial")
            logger.info("Downloading %s...", asset["name"])

            with self._http_client.urlopen(asset["browser_download_url"], timeout=60, headers={"Accept": "application/octet-stream"}) as resp:
                total = int(resp.headers.get("content-length", 0) or 0)
                bytes_done = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_done += len(chunk)
                        if progress_cb:
                            try:
                                progress_cb(bytes_done, total)
                            except Exception:
                                pass

            os.replace(tmp, target)  # atomic on same filesystem
            os.chmod(target, 0o755)

            if not self._validate_llamafile_binary(target):
                try:
                    target.unlink(missing_ok=True)
                finally:
                    raise RuntimeError("Downloaded llamafile failed validation; removed; try again later")

            logger.info("Using existing llamafile: %s", target)
            return str(target.resolve())

    # -----------------------------
    # Launch lifecycle
    # -----------------------------

    @staticmethod
    def _strip_ansi(s: str) -> str:
        import re
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", s)

    def _pump_logs(self):
        """Continuously read child's stdout without blocking startup; store a tail buffer."""
        try:
            if not self.llamafile_process or not self.llamafile_process.stdout:
                return
            for line in iter(self.llamafile_process.stdout.readline, ""):
                clean = self._strip_ansi(line)
                self._log_tail.append(clean.rstrip("\n"))
                logger.debug("%s", clean.rstrip())
        except Exception as e:
            logger.debug("log pump stopped: %s", e)

    @staticmethod
    def _is_probably_binary(first_bytes: bytes) -> bool:
        if len(first_bytes) < 4:
            return False
        magic4 = first_bytes[:4]
        if magic4 == b"\x7fELF":  # ELF
            return True
        # Mach-O and fat
        if magic4 in {b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
                    b"\xca\xfe\xba\xbe", b"\xca\xfe\xd0\x0d", b"\xbe\xba\xfe\xca", b"\x0d\xd0\xfe\xca"}:
            return True
        # Cosmopolitan/APE is a PE with 'MZ' header; accept it
        if first_bytes[:2] == b"MZ":
            return True
        return False

    def _build_argv(self, model_path: str, port: int) -> list[str]:
        argv = [
            self.llamafile_path,
            "--model", model_path,
            "--port", str(port),
        ]
        if logger.isEnabledFor(logging.DEBUG):
            argv.append("--verbose")
        return argv

    def start_llamafile(self, model_path: str, port: int = 8000):
        """Start llamafile subprocess; wait for readiness; stream logs without blocking."""
        if not self.llamafile_path:
            raise RuntimeError("llamafile_path is not set on ModelManager")
        logger.info("Starting llamafile with model: %s on port %s", model_path, port)

        try:
            os.chmod(self.llamafile_path, 0o755)
        except Exception as e:
            logger.debug("chmod failed on llamafile: %s", e)

        argv = self._build_argv(model_path, port)
        env = os.environ.copy()

        def _spawn(argv_list):
            return subprocess.Popen(
                argv_list,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )

        try:
            # Direct exec is correct for ELF, Mach-O, and APE
            self.llamafile_process = _spawn(argv)
        except OSError as e:
            if getattr(e, "errno", None) in {errno.ENOEXEC, 8}:
                # Only use /bin/sh for text scripts with shebang
                sh_argv = ["/bin/sh", self.llamafile_path] + argv[1:]
                self.llamafile_process = _spawn(sh_argv)
            else:
                raise

        atexit.register(self.cleanup)

        if threading.current_thread() == threading.main_thread():
            signal.signal(signal.SIGINT, lambda s, f: self.cleanup())
            signal.signal(signal.SIGTERM, lambda s, f: self.cleanup())

        self._log_thread = threading.Thread(target=self._pump_logs, name="llamafile-log", daemon=True)
        self._log_thread.start()

        logger.info("Waiting for llamafile server to start...")
        health_urls = [f"http://127.0.0.1:{port}/health", f"http://127.0.0.1:{port}/v1/models"]
        deadline = time.time() + 90.0
        last_error = None

        while time.time() < deadline:
            if self.llamafile_process.poll() is not None:
                tail = "\n".join(self._log_tail)
                raise RuntimeError(
                    f"Llamafile process terminated with exit code {self.llamafile_process.returncode}\n"
                    f"---- last log lines ----\n{tail}\n"
                )

            for url in health_urls:
                try:
                    # HTTP; TLS context ignored here; still use wrapper for consistent UA and timeout
                    with self._http_client.urlopen(url, timeout=1.0) as resp:
                        if resp.status == 200:
                            logger.info("Llamafile server is ready!")
                            logger.debug("Llamafile subprocess PID: %s", self.llamafile_process.pid)
                            return
                except Exception as e:
                    last_error = e
            time.sleep(0.5)

        if self.llamafile_process and self.llamafile_process.poll() is None:
            tail = "\n".join(self._log_tail)
            raise RuntimeError(
                "Llamafile server failed to start within 90s\n"
                f"---- last log lines ----\n{tail}\n"
                f"last probe error: {last_error!r}"
            )
        else:
            tail = "\n".join(self._log_tail)
            raise RuntimeError(
                f"Llamafile process exited early with code {self.llamafile_process.returncode}\n"
                f"---- last log lines ----\n{tail}\n"
            )

    def cleanup(self):
        """Clean up llamafile process; terminate process group if present."""
        if self.llamafile_process:
            if self.llamafile_process.poll() is None:
                logger.info("Shutting down llamafile...")
                try:
                    os.killpg(os.getpgid(self.llamafile_process.pid), signal.SIGTERM)
                except Exception:
                    try:
                        self.llamafile_process.terminate()
                    except Exception:
                        pass
                try:
                    self.llamafile_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(self.llamafile_process.pid), signal.SIGKILL)
                    except Exception:
                        try:
                            self.llamafile_process.kill()
                        except Exception:
                            pass
            self.llamafile_process = None