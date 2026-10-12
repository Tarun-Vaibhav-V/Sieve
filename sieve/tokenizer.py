"""Qwen tokenizer wrapper with a flagged character-based fallback.

"total prompt tokens" always come from here; Ollama's prompt_eval_count is
"tokens processed" (it drops when the KV prefix is reused).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Constant chat-template overhead (Qwen ChatML): <|im_start|>role\n ... <|im_end|>\n
DEFAULT_PER_MESSAGE = 5
DEFAULT_PER_REQUEST = 3          # trailing <|im_start|>assistant\n
TOOLS_HEADER_TEXT = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
    "<args-json-object>}\n</tool_call>")


class Tokenizer:
    def __init__(self, model_name: str = "Qwen/Qwen3-14B", force_approx: bool = False,
                 calibration_path: Optional[Path] = None):
        self.model_name = model_name
        self.approx = True
        self._tok = None
        self._memo: "OrderedDict[bytes, int]" = OrderedDict()  # hashed keys, bounded LRU
        self.per_message = DEFAULT_PER_MESSAGE
        self.per_request = DEFAULT_PER_REQUEST
        self.load_error: Optional[str] = None
        if not force_approx:
            try:
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                from transformers import AutoTokenizer  # noqa
                import transformers
                transformers.logging.set_verbosity_error()
                self._tok = AutoTokenizer.from_pretrained(model_name)
                self.approx = False
            except Exception as e:  # pragma: no cover - depends on environment
                self.load_error = f"{type(e).__name__}: {e}"
        if calibration_path and Path(calibration_path).exists():
            try:
                cal = json.loads(Path(calibration_path).read_text()).get("token_calibration", {})
                if cal.get("tokenizer") == model_name and not self.approx:
                    self.per_message = int(cal.get("per_message", self.per_message))
                    self.per_request = int(cal.get("per_request", self.per_request))
            except Exception:
                pass

    def count(self, text: str) -> int:
        if not text:
            return 0
        key = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        hit = self._memo.get(key)
        if hit is not None:
            self._memo.move_to_end(key)
            return hit
        if self._tok is None:
            n = math.ceil(len(text) / 3.6)
        else:
            n = len(self._tok(text, add_special_tokens=False)["input_ids"])
        self._memo[key] = n
        if len(self._memo) > 100_000:
            self._memo.popitem(last=False)
        return n

    def tools_tokens(self, tools: Optional[List[Dict[str, Any]]]) -> int:
        if not tools:
            return 0
        body = "\n".join(json.dumps({"type": "function", "function": t["function"]}) for t in tools)
        return self.count(TOOLS_HEADER_TEXT) + self.count(body)

    def count_messages(self, messages: List[Dict[str, Any]], tools=None) -> int:
        total = self.per_request + self.tools_tokens(tools)
        for m in messages:
            total += self.per_message + self.count(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                total += self.count(json.dumps(tc.get("function", {}))) + 4
        return total

    def truncate_to(self, text: str, n: int) -> str:
        """Cut text to at most n tokens (used only for expansion budgets)."""
        if self.count(text) <= n:
            return text
        if self._tok is None:
            return text[: int(n * 3.6)]
        ids = self._tok(text, add_special_tokens=False)["input_ids"][:n]
        return self._tok.decode(ids)


_SINGLETON: Dict[str, Tokenizer] = {}


def get_tokenizer(model_name: str = "Qwen/Qwen3-14B", force_approx: bool = False,
                  calibration_path: Optional[Path] = None) -> Tokenizer:
    key = f"{model_name}|{force_approx}"
    if key not in _SINGLETON:
        _SINGLETON[key] = Tokenizer(model_name, force_approx, calibration_path)
    return _SINGLETON[key]
