"""On-device inference through llama.cpp.

All model computation happens on this machine: the ``llama`` binary is started
as a local child process bound to ``127.0.0.1`` and the model weights are read
from a local GGUF file.  No request ever leaves the host, and no external
inference API is used.

Two capabilities are used by the agent:

* ``complete`` - plain text completion, optionally constrained by a GBNF
  grammar so that a small model can only emit syntactically valid output.
* ``complete_logprobs`` - the model's token-level log-probabilities, which turn
  the model into a genuine *distribution* over the legal moves rather than a
  single sampled answer.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["LlamaCppConfig", "LlamaCppServer", "find_llama_binary", "free_port"]


def find_llama_binary(explicit: str | None = None) -> str:
    """Locate the ``llama`` executable."""
    candidates = [explicit] if explicit else []
    candidates += [os.environ.get("LLAMA_BIN", ""), str(Path.home() / ".local" / "bin" / "llama"), "llama"]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise FileNotFoundError(
        "could not find the 'llama' binary. Install llama.cpp "
        "(curl -LsSf https://llama.app/install.sh | sh) or set LLAMA_BIN."
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class LlamaCppConfig:
    model_path: Path
    binary: str = ""
    host: str = "127.0.0.1"
    port: int = 0
    n_predict: int = 96
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: int = 0
    n_threads: int = 0
    n_gpu_layers: int = 0
    context_size: int = 4096
    startup_timeout: float = 180.0
    extra_args: list[str] = field(default_factory=list)


class LlamaCppServer:
    """A local llama.cpp server process, used as a completion backend."""

    def __init__(self, config: LlamaCppConfig):
        self.config = config
        self.process: subprocess.Popen | None = None
        self._port = config.port or free_port()

    @property
    def endpoint(self) -> str:
        return f"http://{self.config.host}:{self._port}"

    def command(self) -> list[str]:
        cfg = self.config
        cmd = [
            find_llama_binary(cfg.binary or None),
            "server",
            "-m",
            str(cfg.model_path),
            "--host",
            cfg.host,
            "--port",
            str(self._port),
            "-c",
            str(cfg.context_size),
            "-t",
            str(cfg.n_threads or os.cpu_count() or 4),
            "-ngl",
            str(cfg.n_gpu_layers),
        ]
        return cmd + list(cfg.extra_args)

    def start(self) -> LlamaCppServer:
        cfg = self.config
        if not Path(cfg.model_path).is_file():
            raise FileNotFoundError(f"model file not found: {cfg.model_path}")
        self.process = subprocess.Popen(self.command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._wait_until_healthy()
        return self

    def _wait_until_healthy(self) -> None:
        deadline = time.time() + self.config.startup_timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"llama.cpp server exited with code {self.process.returncode} while loading "
                    f"{self.config.model_path}"
                )
            try:
                with urllib.request.urlopen(f"{self.endpoint}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.4)
        raise TimeoutError("llama.cpp server did not become healthy in time")

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.endpoint}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete(self, prompt: str, n_predict: int | None = None, stop: list[str] | None = None, grammar: str | None = None) -> str:
        """Run a raw completion and return the generated text.

        ``grammar`` is a GBNF grammar; with one in place the model can only
        produce output that satisfies it, which keeps a small model honest.
        """
        cfg = self.config
        payload = {
            "prompt": prompt,
            "n_predict": n_predict if n_predict is not None else cfg.n_predict,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repeat_penalty": cfg.repeat_penalty,
            "seed": cfg.seed,
            "stop": stop or [],
            "cache_prompt": True,
        }
        if grammar:
            payload["grammar"] = grammar
        return str(self._post(payload).get("content", ""))

    def complete_timed(
        self,
        prompt: str,
        n_predict: int | None = None,
        stop: list[str] | None = None,
        grammar: str | None = None,
        cache_prompt: bool = True,
    ) -> dict:
        """Like :meth:`complete`, but also returns llama.cpp's timing report."""
        cfg = self.config
        payload = {
            "prompt": prompt,
            "n_predict": n_predict if n_predict is not None else cfg.n_predict,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repeat_penalty": cfg.repeat_penalty,
            "seed": cfg.seed,
            "stop": stop or [],
            "cache_prompt": cache_prompt,
        }
        if grammar:
            payload["grammar"] = grammar
        body = self._post(payload)
        return {"content": str(body.get("content", "")), "timings": body.get("timings", {})}

    def complete_logprobs(self, prompt: str, n_predict: int = 1, n_probs: int = 20, temperature: float = 1.0) -> list[dict]:
        """Return the top-``n_probs`` log-probabilities for each generated position.

        Each entry is ``{"token": str, "logprob": float}``.
        """
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": n_probs,
            "repeat_penalty": 1.0,
            "n_probs": n_probs,
            "cache_prompt": True,
        }
        body = self._post(payload)
        result = []
        for position in body.get("completion_probabilities") or []:
            result.append(
                {
                    "token": str(entry.get("token", "")),
                    "logprob": float(entry.get("logprob", 0.0)),
                }
                for entry in position.get("top_logprobs") or []
            )
        return result

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)
        self.process = None

    def __enter__(self) -> LlamaCppServer:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()
