"""Config dataclass loaded from config.yaml, overridable from the CLI."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    model: str = "qwen3:14b"
    ollama_host: str = "http://localhost:11434"
    num_ctx: int = 4096
    num_predict: int = 64
    temperature: float = 0
    seed: int = 0
    think: bool = False
    keep_alive: str = "30m"
    reserve_output_tokens: int = 256
    budget_fraction: float = 0.80
    epoch_trigger_fraction: float = 0.80
    epoch_hysteresis: float = 0.85
    compressor_model: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    tokenizer_model: str = "Qwen/Qwen3-14B"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ratio_buckets: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9, 1.0])
    min_ratio_by_kind: Dict[str, float] = field(default_factory=lambda: {
        "system": 1.0, "instruction": 0.9, "doc": 0.3, "chat": 0.4,
        "tool_result": 0.3, "attachment": 0.3, "query": 1.0})
    min_segment_tokens: int = 40
    tool_mode: str = "hybrid"
    prefetch_sim_threshold: float = 0.75
    expansion_budget_tokens: int = 1200
    max_tool_rounds: int = 2
    classifier_conf_threshold: float = 0.55
    cost_matrix: Dict[str, float] = field(default_factory=lambda: {
        "needed_externalized": 5, "needless_compress_of_externalizable": 1,
        "cache_confusion": 1, "default": 1})
    summary_mode: str = "extractive"
    verifier_min_recall: float = 1.0
    verifier_overall_min_recall: float = 0.9
    verifier_min_context_recall: float = 0.5
    verifier_max_retries: int = 3
    pynvml_sample_ms: int = 200
    chunk_min_tokens: int = 150
    chunk_max_tokens: int = 400
    neardup_jaccard: float = 0.85
    classifier_labels: str = "oracle"
    train_seeds: List[int] = field(default_factory=lambda: [1001, 1002, 1003, 1004, 1005, 1006])
    extra_rules: List[Dict[str, Any]] = field(default_factory=list)
    # runtime-only (not in yaml)
    mock: bool = False
    runs_dir: str = str(ROOT / "runs")

    @property
    def budget_limit(self) -> int:
        """Hard prompt-token limit: budget share of num_ctx, never eating the output reserve."""
        return int(min(self.budget_fraction * self.num_ctx, self.num_ctx - self.reserve_output_tokens))

    @property
    def epoch_trigger_tokens(self) -> int:
        """Prompt tokens above which (prompt + reserve) crosses epoch_trigger_fraction*num_ctx."""
        return int(self.epoch_trigger_fraction * self.num_ctx - self.reserve_output_tokens)

    def ollama_options(self, num_ctx: int | None = None) -> Dict[str, Any]:
        return {"num_ctx": num_ctx or self.num_ctx, "num_predict": self.num_predict,
                "temperature": self.temperature, "seed": self.seed}

    def snap_ratio(self, r: float, kind: str | None = None) -> float:
        b = min(self.ratio_buckets, key=lambda x: (abs(x - r), -x))
        if kind is not None:
            floor = self.min_ratio_by_kind.get(kind, 0.3)
            if b < floor:
                b = min([x for x in self.ratio_buckets if x >= floor] or [1.0])
        return b

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(cur: Any, val: str) -> Any:
    if isinstance(cur, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(val)
    if isinstance(cur, float):
        return float(val)
    if isinstance(cur, (list, dict)):
        return json.loads(val)
    return val


def load_config(path: str | Path | None = None, overrides: Dict[str, str] | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in (data or {}).items() if k in known})
    for k, v in (overrides or {}).items():
        if k not in known:
            raise SystemExit(f"Unknown config key: {k}")
        setattr(cfg, k, _coerce(getattr(cfg, k), v) if isinstance(v, str) else v)
    return cfg
