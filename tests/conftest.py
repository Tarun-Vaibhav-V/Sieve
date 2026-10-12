import os
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    from sieve.config import load_config
    c = load_config()
    c.mock = True
    c.runs_dir = str(tmp_path_factory.mktemp("runs"))
    return c


@pytest.fixture(scope="session")
def rt(cfg):
    """Fast runtime: real Qwen tokenizer if cached, TF-IDF similarity, extractive fallback compressor."""
    from sieve.conditions import Runtime
    return Runtime(cfg, force_tfidf=True, force_fallback_compressor=True)


@pytest.fixture(scope="session")
def trained(cfg, rt):
    """Small classifier trained on two seeds of the smoke tier (enough for pipeline tests)."""
    from sieve.classifier import LaneClassifier, build_rows, label_rows, model_path
    from sieve.features import ALL_FEATURES, CORE_FEATURES
    from sieve.workloads.generator import generate_suite
    cases = [c for s in (1001, 1002) for c in generate_suite("smoke", s)]
    rows = build_rows(cases, cfg, rt.tok, rt.sim)
    for labels in ("oracle", "heuristic"):
        y, r = label_rows(rows, labels, cfg)
        LaneClassifier(cfg, ALL_FEATURES, labels).fit(rows, y, r).save(model_path(cfg, labels, "all"))
    y, r = label_rows(rows, "oracle", cfg)
    LaneClassifier(cfg, CORE_FEATURES, "oracle").fit(rows, y, r).save(model_path(cfg, "oracle", "core6"))
    return rows
