"""Ollama native /api/chat client: streaming, timing, /api/ps, tool loop."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
NS = 1e9


class OllamaUnavailable(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    thinking: str = ""
    thinking_leak: bool = False
    ttft_s: Optional[float] = None
    total_s: float = 0.0
    prompt_eval_count: int = 0
    prompt_eval_s: float = 0.0
    eval_count: int = 0
    eval_s: float = 0.0
    load_s: float = 0.0
    total_duration_s: float = 0.0
    decode_tps: float = 0.0
    vram_peak_mb: Optional[float] = None
    vram_source: str = "none"
    done_reason: str = ""
    raw_final: Dict[str, Any] = field(default_factory=dict)

    def metrics(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "ttft_s", "total_s", "prompt_eval_count", "prompt_eval_s", "eval_count", "eval_s",
            "load_s", "decode_tps", "vram_peak_mb", "thinking_leak", "done_reason")}


class VramSampler:
    """Peak GPU memory during a request: pynvml -> nvidia-smi -> unavailable."""

    def __init__(self, interval_ms: int = 200):
        self.interval = interval_ms / 1000.0
        self.peak: Optional[float] = None
        self.source = "none"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
            self.source = "pynvml"
        except Exception:
            if shutil.which("nvidia-smi"):
                self.source = "nvidia-smi"

    def _read(self) -> Optional[float]:
        try:
            if self._nvml is not None:
                return self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml).used / 2**20
            if self.source == "nvidia-smi":
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=2).stdout
                return float(out.strip().splitlines()[0])
        except Exception:
            return None
        return None

    def _loop(self):
        while not self._stop.is_set():
            v = self._read()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.source != "none":
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


class OllamaClient:
    is_mock = False

    def __init__(self, cfg, tokenizer=None, retries: int = 3):
        self.cfg = cfg
        self.host = cfg.ollama_host.rstrip("/")
        self.retries = retries
        self.tok = tokenizer
        self.http = httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0))
        self._ps_cache: Dict[str, Any] = {}

    # ---------- low level ----------
    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        last = None
        for attempt in range(self.retries):
            try:
                r = self.http.request(method, self.host + path, **kw)
                r.raise_for_status()
                return r
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise OllamaUnavailable(
            f"Cannot reach Ollama at {self.host} ({last}). Start it with `ollama serve` "
            f"(recommended env: OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 "
            f"OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1).")

    def version(self) -> str:
        return self._request("GET", "/api/version").json().get("version", "?")

    def list_models(self) -> List[str]:
        return [m["name"] for m in self._request("GET", "/api/tags").json().get("models", [])]

    def show(self, model: Optional[str] = None) -> Dict[str, Any]:
        return self._request("POST", "/api/show", json={"model": model or self.cfg.model}).json()

    def ps(self) -> Dict[str, Any]:
        """size / size_vram of the loaded model; gpu_pct = size_vram/size."""
        for m in self._request("GET", "/api/ps").json().get("models", []):
            if m.get("name") == self.cfg.model or m.get("model") == self.cfg.model:
                size, vram = m.get("size", 0), m.get("size_vram", 0)
                self._ps_cache = {"size": size, "size_vram": vram, "gpu_pct": (vram / size) if size else 0.0,
                                  "context_length": m.get("context_length")}
                return self._ps_cache
        return {"size": 0, "size_vram": 0, "gpu_pct": 0.0}

    # ---------- chat ----------
    def chat(self, messages: List[Dict[str, Any]], tools=None, options=None,
             num_ctx: Optional[int] = None, keep_alive: Optional[str] = None) -> ChatResult:
        opts = dict(self.cfg.ollama_options(num_ctx))
        if options:
            opts.update(options)
        payload = {"model": self.cfg.model, "messages": messages, "stream": True,
                   "think": self.cfg.think, "options": opts,
                   "keep_alive": keep_alive if keep_alive is not None else self.cfg.keep_alive}
        if tools:
            payload["tools"] = tools
        res = ChatResult()
        content, thinking, calls = [], [], []
        final: Dict[str, Any] = {}
        sampler = VramSampler(self.cfg.pynvml_sample_ms)
        t0 = time.perf_counter()
        for attempt in range(self.retries):
            try:
                with sampler, self.http.stream("POST", self.host + "/api/chat", json=payload) as r:
                    if r.status_code >= 400:
                        body = r.read().decode(errors="replace")
                        raise RuntimeError(f"Ollama error {r.status_code}: {body[:400]}")
                    for line in r.iter_lines():
                        if not line:
                            continue
                        ch = json.loads(line)
                        if "error" in ch:
                            raise RuntimeError(f"Ollama error: {ch['error']}")
                        msg = ch.get("message") or {}
                        if msg.get("thinking"):
                            thinking.append(msg["thinking"])
                        if msg.get("content") or msg.get("tool_calls"):
                            if res.ttft_s is None:
                                res.ttft_s = time.perf_counter() - t0
                            if msg.get("content"):
                                content.append(msg["content"])
                            calls.extend(msg.get("tool_calls") or [])
                        if ch.get("done"):
                            final = ch
                break
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
                if attempt == self.retries - 1:
                    raise OllamaUnavailable(f"Ollama connection failed during chat: {e}")
                time.sleep(2 * (attempt + 1))
                content, thinking, calls, res.ttft_s, t0 = [], [], [], None, time.perf_counter()
        res.total_s = time.perf_counter() - t0
        text = "".join(content)
        th = "".join(thinking)
        if THINK_RE.search(text) or "<think>" in text:
            th += " ".join(THINK_RE.findall(text))
            text = THINK_RE.sub("", text).replace("<think>", "").replace("</think>", "")
        res.content = text.strip()
        res.thinking = th.strip()
        res.tool_calls = calls
        res.raw_final = {k: v for k, v in final.items() if k != "message"}
        res.prompt_eval_count = int(final.get("prompt_eval_count", 0) or 0)
        res.prompt_eval_s = (final.get("prompt_eval_duration", 0) or 0) / NS
        res.eval_count = int(final.get("eval_count", 0) or 0)
        res.eval_s = (final.get("eval_duration", 0) or 0) / NS
        res.load_s = (final.get("load_duration", 0) or 0) / NS
        res.total_duration_s = (final.get("total_duration", 0) or 0) / NS
        res.decode_tps = res.eval_count / res.eval_s if res.eval_s > 0 else 0.0
        res.done_reason = final.get("done_reason", "")
        answer_toks = self.tok.count(res.content) if self.tok else len(res.content) / 3.6
        call_toks = sum(len(json.dumps(c)) / 3.0 for c in calls)
        res.thinking_leak = bool(res.thinking) or (res.eval_count > 3 * (answer_toks + call_toks) + 24)
        if sampler.peak is not None:
            res.vram_peak_mb, res.vram_source = sampler.peak, sampler.source
        else:  # Apple silicon / no NVIDIA: model's resident GPU share from /api/ps
            try:
                p = self.ps()
                res.vram_peak_mb, res.vram_source = p.get("size_vram", 0) / 2**20, "ollama_ps(size_vram)"
            except Exception:
                pass
        return res

    def warmup(self, num_ctx: Optional[int] = None) -> ChatResult:
        return self.chat([{"role": "user", "content": "Say OK."}], options={"num_predict": 2}, num_ctx=num_ctx)

    def flush_prefix(self, num_ctx: Optional[int] = None) -> None:
        """Short unrelated prompt so the next request shares no KV prefix."""
        self.chat([{"role": "user", "content": "Reply with one word: ready."}], options={"num_predict": 1},
                  num_ctx=num_ctx)

    def unload(self) -> None:
        try:
            self._request("POST", "/api/chat", json={"model": self.cfg.model, "messages": [], "keep_alive": 0})
        except Exception:
            pass
        for _ in range(30):  # wait until /api/ps no longer lists it
            if not self.ps().get("size"):
                return
            time.sleep(0.5)

    # ---------- tool loop ----------
    def run_with_tools(self, messages: List[Dict[str, Any]], tools, expand_fn: Callable[[str], str],
                       max_rounds: int, num_ctx: Optional[int] = None) -> Dict[str, Any]:
        """Call; if the model calls expand_context, append the call and the tool result at the
        tail of the message list and call again (up to max_rounds tool rounds)."""
        msgs = list(messages)
        rounds: List[ChatResult] = []
        expansions: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        tool_rounds = 0
        while True:
            r = self.chat(msgs, tools=tools, num_ctx=num_ctx)
            rounds.append(r)
            calls = [c for c in r.tool_calls if (c.get("function") or {}).get("name") == "expand_context"]
            if not calls or not tools or tool_rounds >= max_rounds:
                break
            tool_rounds += 1
            msgs.append({"role": "assistant", "content": r.content or "", "tool_calls": calls})
            for c in calls:
                args = (c.get("function") or {}).get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"seg_id": args}
                sid = normalize_seg_id(args)
                out = expand_fn(sid)
                expansions.append({"seg": sid, "round": tool_rounds})
                msgs.append({"role": "tool", "content": out, "tool_name": "expand_context"})
        final = rounds[-1]
        first_content_ttft = None
        elapsed = 0.0
        for r in rounds:
            if r is final:
                first_content_ttft = elapsed + (r.ttft_s or r.total_s)
            elapsed += r.total_s
        return {
            "answer": final.content if not (final.tool_calls and not final.content) else final.content,
            "unanswered_tool_call": bool(final.tool_calls) and not final.content,
            "rounds": rounds, "tool_rounds": tool_rounds, "expansions": expansions, "messages": msgs,
            "ttft_s": first_content_ttft, "total_s": time.perf_counter() - t0,
            "prompt_eval_count": sum(r.prompt_eval_count for r in rounds),
            "first_prompt_eval_count": rounds[0].prompt_eval_count,
            "prompt_eval_s": sum(r.prompt_eval_s for r in rounds),
            "eval_count": sum(r.eval_count for r in rounds),
            "eval_s": sum(r.eval_s for r in rounds),
            "load_s": sum(r.load_s for r in rounds),
            "decode_tps": (sum(r.eval_count for r in rounds) / sum(r.eval_s for r in rounds))
            if sum(r.eval_s for r in rounds) > 0 else 0.0,
            "vram_peak_mb": max([r.vram_peak_mb or 0 for r in rounds]) or None,
            "vram_source": rounds[0].vram_source,
            "thinking_leak": any(r.thinking_leak for r in rounds),
        }


SEG_RE = re.compile(r"[0-9a-f]{12}")


def normalize_seg_id(args: Dict[str, Any]) -> str:
    v = None
    for k in ("seg_id", "id", "segment_id", "ref", "seg"):
        if k in args:
            v = str(args[k])
            break
    if v is None and args:
        v = str(next(iter(args.values())))
    v = v or ""
    m = SEG_RE.search(v)
    return m.group(0) if m else v.strip()
