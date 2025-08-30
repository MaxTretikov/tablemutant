#!/usr/bin/env python3
import subprocess
import sys
import os
import platform
import signal
import atexit
import time
import threading
import urllib.request
import urllib.parse
import json
import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from collections import deque

# Get logger for this module
logger = logging.getLogger('tablemutant.core.model_manager')


class ModelManager:
    def __init__(self):
        self.llamafile_process = None
        self.llamafile_path = None
        self.model_path = None
        self._log_tail = deque(maxlen=400)
        self._log_thread = None

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
            system = platform.system()
            logger.debug("get_models_dir system: %s", system)
            if system == "Linux":
                models_dir = Path.home() / ".tablemutant" / "models"
            elif system == "Darwin":
                models_dir = Path.home() / "Library" / "Application Support" / "TableMutant" / "models"
            elif system == "Windows":
                models_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "TableMutant" / "models"
            else:
                models_dir = Path.home() / ".tablemutant" / "models"

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

                print(f"Downloading {filename} from {repo_id}...")
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=models_dir,
                    local_dir=local_repo_dir,
                )
                return str(Path(local_path).resolve())

            except ImportError:
                print("huggingface-hub not installed. Install with: pip install huggingface-hub")
                sys.exit(1)
            except Exception as e:
                print(f"Error downloading from HuggingFace: {e}")
                sys.exit(1)

        elif model_identifier.startswith("http"):
            filename = os.path.basename(urllib.parse.urlparse(model_identifier).path)
            local_path = models_dir / filename

            if local_path.exists():
                print(f"Model already downloaded: {local_path}")
                return str(local_path)

            print(f"Downloading {model_identifier}...")

            with urllib.request.urlopen(model_identifier) as response:
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

    def download_llamafile(self) -> str:
        """Download the latest llamafile if not present."""
        llamafile_dir = Path.home() / ".tablemutant" / "bin"
        logger.debug("download_llamafile llamafile_dir: %s", llamafile_dir)
        llamafile_dir.mkdir(parents=True, exist_ok=True)

        existing_llamafiles = list(llamafile_dir.glob("llamafile-*"))
        logger.debug("download_llamafile existing_llamafiles: %s", existing_llamafiles)
        if existing_llamafiles:
            llamafile_path = max(existing_llamafiles, key=os.path.getctime)
            print(f"Using existing llamafile: {llamafile_path}")
            os.chmod(llamafile_path, 0o755)
            return str(llamafile_path)

        print("Fetching latest llamafile release...")
        with urllib.request.urlopen("https://api.github.com/repos/Mozilla-Ocho/llamafile/releases/latest") as response:
            release_data = json.loads(response.read().decode())

        llamafile_asset = None
        for asset in release_data["assets"]:
            name = asset["name"]
            if name.startswith("llamafile-") and not any(ext in name for ext in [".zip", ".tar", ".exe", ".gz"]):
                llamafile_asset = asset
                break

        if not llamafile_asset:
            raise ValueError("Could not find llamafile asset in latest release")

        local_path = llamafile_dir / llamafile_asset["name"]
        print(f"Downloading {llamafile_asset['name']}...")

        with urllib.request.urlopen(llamafile_asset["browser_download_url"]) as response:
            total_size = int(response.headers.get("content-length", 0))
            with open(local_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        os.chmod(local_path, 0o755)
        return str(local_path)

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
                print(clean, end="")
        except Exception as e:
            logger.debug("log pump stopped: %s", e)

    @staticmethod
    def _is_probably_binary(first_bytes: bytes) -> bool:
        """Detect Mach-O or ELF quickly to decide whether we can exec directly."""
        if len(first_bytes) < 4:
            return False
        magic4 = first_bytes[:4]
        # ELF
        if magic4 == b"\x7fELF":
            return True
        # Mach-O thin
        if magic4 in {b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe"}:
            return True
        # Mach-O fat
        if magic4 in {b"\xca\xfe\xba\xbe", b"\xca\xfe\xd0\x0d", b"\xbe\xba\xfe\xca", b"\x0d\xd0\xfe\xca"}:
            return True
        return False

    def _build_argv(self, model_path: str, port: int) -> list[str]:
        argv = [
            self.llamafile_path,
            "--model", model_path,
            "--port", str(port),
        ]
        
        # Only add --verbose if log level is DEBUG
        if logger.isEnabledFor(logging.DEBUG):
            argv.append("--verbose")
            
        return argv

    def start_llamafile(self, model_path: str, port: int = 8000):
        """Start llamafile subprocess; wait for readiness; stream logs without blocking."""
        if not self.llamafile_path:
            raise RuntimeError("llamafile_path is not set on ModelManager")
        print(f"Starting llamafile with model: {model_path} on port {port}")

        # Ensure executable bit
        try:
            os.chmod(self.llamafile_path, 0o755)
        except Exception as e:
            logger.debug("chmod failed on llamafile: %s", e)

        argv = self._build_argv(model_path, port)

        # Prepare environment; keep it mostly inherited; no special vars required
        env = os.environ.copy()

        # Launch without shell; keep stdin open; detach session; merge stderr into stdout
        # Try a direct exec first; on Exec format error, fall back to /bin/sh wrapper
        try:
            with open(self.llamafile_path, "rb") as f:
                head = f.read(4096)
            direct_ok = head.startswith(b"#!") or self._is_probably_binary(head)
        except Exception:
            direct_ok = False

        def _spawn(argv_list):
            return subprocess.Popen(
                argv_list,
                stdin=subprocess.PIPE,            # keep stdin open; avoid EOF-triggered shutdowns
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,           # separate process group
            )

        try:
            if direct_ok:
                self.llamafile_process = _spawn(argv)
            else:
                # No recognizable magic; run under /bin/sh explicitly
                sh_argv = ["/bin/sh", self.llamafile_path] + argv[1:]
                self.llamafile_process = _spawn(sh_argv)
        except OSError as e:
            # Fallback if the kernel reports Exec format error anyway
            if getattr(e, "errno", None) == 8:
                logger.debug("direct exec failed with Exec format error; retrying via /bin/sh")
                sh_argv = ["/bin/sh", self.llamafile_path] + argv[1:]
                self.llamafile_process = _spawn(sh_argv)
            else:
                raise

        # Register cleanup
        atexit.register(self.cleanup)

        # Install signal handlers only in the main thread
        if threading.current_thread() == threading.main_thread():
            signal.signal(signal.SIGINT, lambda s, f: self.cleanup())
            signal.signal(signal.SIGTERM, lambda s, f: self.cleanup())

        # Start background log pump
        self._log_thread = threading.Thread(target=self._pump_logs, name="llamafile-log", daemon=True)
        self._log_thread.start()

        # Readiness probe
        print("Waiting for llamafile server to start...")
        health_urls = [f"http://127.0.0.1:{port}/health", f"http://127.0.0.1:{port}/v1/models"]
        deadline = time.time() + 90.0
        last_error = None

        while time.time() < deadline:
            # If process exited during startup, surface logs
            if self.llamafile_process.poll() is not None:
                tail = "\n".join(self._log_tail)
                raise RuntimeError(
                    f"Llamafile process terminated with exit code {self.llamafile_process.returncode}\n"
                    f"---- last log lines ----\n{tail}\n"
                )

            for url in health_urls:
                try:
                    with urllib.request.urlopen(url, timeout=1.0) as resp:
                        if resp.status == 200:
                            print("Llamafile server is ready!")
                            logger.debug("Llamafile subprocess PID: %s", self.llamafile_process.pid)
                            return
                except Exception as e:
                    last_error = e
                    # ignore; retry
            time.sleep(0.5)

        # Timed out
        if self.llamafile_process and self.llamafile_process.poll() is None:
            # Still running; include tail to help debug
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
                print("\nShutting down llamafile...")
                try:
                    # Terminate the whole group created via start_new_session
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