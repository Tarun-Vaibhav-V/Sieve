"""Lane classifier (GradientBoosting) + ratio regressor (RandomForest), CPU, ms inference.

Decisions minimise expected cost under cfg.cost_matrix rather than taking the argmax; if the
max probability is below classifier_conf_threshold we fall back to COMPRESS@0.9 (never
EXTERNALIZE). Training data is split by conversation (GroupKFold), never by segment.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from sieve.features import ALL_FEATURES, CORE_FEATURES, compute_features, vectorize
from sieve.rules import LANES, Decision, apply_rules, heuristic_label
from sieve.segmenter import Segment, SessionState


def cost_matrix(cfg) -> np.ndarray:
    """C[true, pred] over LANES = [cache, compress, externalize]."""
    cm = cfg.cost_matrix
    d = cm.get("default", 1)
    C = np.full((3, 3), float(d))
    np.fill_diagonal(C, 0.0)
    ci, co, ex = LANES.index("cache"), LANES.index("compress"), LANES.index("externalize")
    C[co, ex] = cm.get("needed_externalized", 5)            # needed content hidden behind a stub
    C[ex, co] = cm.get("needless_compress_of_externalizable", 1)
    for j in (co, ex):
        C[ci, j] = C[j, ci] = cm.get("cache_confusion", 1)
    return C


def oracle_label(seg: Segment, needed: bool, cfg) -> Tuple[str, float]:
    f = seg.features
    if f.get("is_system") or seg.tags.get("static") or seg.tags.get("pinned") or f.get("repeat_count", 0) > 0:
        return "cache", 1.0
    if needed:
        return "compress", 1.0 if f.get("digit_density", 0) > 0.03 else 0.9
    if f.get("turns_old", 0) >= 6 and seg.tokens >= 80:
        return "externalize", 1.0
    q = f.get("query_similarity", 0)
    return "compress", cfg.snap_ratio(0.3 + 0.6 * min(1.0, q / 0.6), seg.kind)


def segment_needed(seg: Segment, case: Dict[str, Any], from_eval: int) -> bool:
    for ep in case["eval_points"][from_eval:]:
        for fact in ep.get("facts") or []:
            if fact and fact in seg.content:
                return True
    return False


def build_rows(cases: List[Dict[str, Any]], cfg, tok, sim) -> List[Dict[str, Any]]:
    """One row per (eval point, history segment) using the features the router would see."""
    rows = []
    for case in cases:
        st = SessionState(case["case_id"], tok, cfg)
        for k, ep in enumerate(case["eval_points"]):
            qi = ep["question_index"]
            msgs = [dict(m) for m in case["messages"][:qi]] + [dict(case["messages"][qi], kind="query")]
            segs = st.segment(msgs)
            compute_features(segs, msgs[-1]["content"], int(msgs[-1].get("turn", 0)), sim)
            lanes = {}
            for s in segs:
                if s.kind == "query":
                    continue
                rr = apply_rules(s, cfg, lanes)
                lanes[s.key] = "compress"
                rows.append(dict(group=f"{case['seed']}:{case['case_id']}", case_id=case["case_id"],
                                 family=case["family"], eval_point=k, seg=s, x=vectorize(s),
                                 needed=segment_needed(s, case, k),
                                 rule=(rr["decision"].source if rr["decision"] else None),
                                 rule_lane=(rr["decision"].lane if rr["decision"] else None),
                                 rule_ratio=(rr["decision"].ratio if rr["decision"] else None),
                                 forbid=rr["forbid_externalize"]))
    return rows


def label_rows(rows, labels: str, cfg, noise: float = 0.1, seed: int = 0):
    rng = random.Random(seed)
    ys, rs = [], []
    for r in rows:
        s = r["seg"]
        if labels == "oracle":
            lane, ratio = oracle_label(s, r["needed"], cfg)
        else:  # heuristic bootstrap, with label noise
            d = heuristic_label(s, cfg)
            lane, ratio = d.lane, d.ratio
            if rng.random() < noise:
                lane = rng.choice(LANES)
            if lane == "compress" and rng.random() < noise:
                b = cfg.ratio_buckets
                ratio = b[max(0, min(len(b) - 1, b.index(ratio) + rng.choice([-1, 1])))] if ratio in b else ratio
        ys.append(lane)
        rs.append(ratio)
    return ys, rs


class LaneClassifier:
    def __init__(self, cfg, features: List[str] = ALL_FEATURES, labels: str = "oracle"):
        self.cfg = cfg
        self.features = list(features)
        self.labels = labels
        self.lane_model = None
        self.ratio_model = None
        self.mu = None
        self.sd = None
        self.C = cost_matrix(cfg)

    def _X(self, rows) -> np.ndarray:
        idx = [ALL_FEATURES.index(f) for f in self.features]
        return np.array([[r["x"][i] for i in idx] for r in rows], dtype=float)

    def fit(self, rows, ys, rs, seed: int = 0):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
        X = self._X(rows)
        y = np.array([LANES.index(v) for v in ys])
        counts = np.bincount(y, minlength=3).astype(float)
        w = np.array([len(y) / (3 * max(1.0, counts[v])) for v in y])  # class_weight="balanced"
        self.lane_model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed)
        self.lane_model.fit(X, y, sample_weight=w)
        comp = y == LANES.index("compress")
        self.ratio_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=seed, n_jobs=1)
        self.ratio_model.fit(X[comp], np.array(rs)[comp])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        p = self.lane_model.predict_proba(X)
        full = np.zeros((len(X), 3))
        for j, c in enumerate(self.lane_model.classes_):
            full[:, c] = p[:, j]
        return full

    def decide_from(self, p: np.ndarray, ratio_raw: float, kind: str) -> Tuple[str, float, float, str]:
        exp_cost = p @ self.C                     # expected cost of predicting each lane
        j = int(np.argmin(exp_cost))
        conf = float(p.max())
        if conf < self.cfg.classifier_conf_threshold:
            return "compress", self.cfg.snap_ratio(0.9, kind), float(exp_cost[LANES.index("compress")]), "fallback"
        lane = LANES[j]
        ratio = self.cfg.snap_ratio(ratio_raw, kind) if lane == "compress" else 1.0
        return lane, ratio, float(exp_cost[j]), "classifier"

    def predict(self, seg: Segment) -> Decision:
        x = self._X([{"x": vectorize(seg)}])
        p = self.proba(x)[0]
        r = float(self.ratio_model.predict(x)[0])
        lane, ratio, ec, src = self.decide_from(p, r, seg.kind)
        imp = self.lane_model.feature_importances_
        z = (x[0] - self.mu) / self.sd
        contrib = sorted(zip(self.features, imp * z), key=lambda t: -abs(t[1]))[:3]
        return Decision(seg.id, seg.key, src, lane, ratio,
                        probs={LANES[i]: round(float(p[i]), 3) for i in range(3)}, confidence=round(float(p.max()), 3),
                        expected_cost=round(ec, 3), reasons=[f"{n}:{v:+.2f}" for n, v in contrib])

    def predict_rows(self, rows) -> Tuple[List[str], List[float], np.ndarray]:
        X = self._X(rows)
        P = self.proba(X)
        R = self.ratio_model.predict(X)
        lanes, ratios = [], []
        for r, p, rr in zip(rows, P, R):
            lane, ratio, _, _ = self.decide_from(p, rr, r["seg"].kind)
            lanes.append(lane)
            ratios.append(ratio)
        return lanes, ratios, P

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"features": self.features, "labels": self.labels, "lane": self.lane_model,
                     "ratio": self.ratio_model, "mu": self.mu, "sd": self.sd}, path)

    @classmethod
    def load(cls, path: Path, cfg) -> "LaneClassifier":
        d = joblib.load(path)
        c = cls(cfg, d["features"], d["labels"])
        c.lane_model, c.ratio_model, c.mu, c.sd = d["lane"], d["ratio"], d["mu"], d["sd"]
        return c


def model_path(cfg, labels: str, features: str = "all") -> Path:
    return Path(cfg.runs_dir) / "models" / f"lane_{labels}_{features}.joblib"


def rules_only_route(row, cfg) -> Tuple[str, float]:
    s = row["seg"]
    if row["rule"]:
        return row["rule_lane"], row["rule_ratio"]
    return "compress", cfg.snap_ratio(0.9, s.kind)


def evaluate(rows, cfg, n_splits: int = 5, seed: int = 0) -> Dict[str, Any]:
    """GroupKFold by conversation. Ground truth = oracle labels. Compares rules-only,
    heuristic labeller, heuristic-trained classifier, oracle-trained classifier, and a
    core-six-feature ablation."""
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
    from sklearn.model_selection import GroupKFold
    y_or, r_or = label_rows(rows, "oracle", cfg)
    y_he, r_he = label_rows(rows, "heuristic", cfg, seed=seed)
    groups = [r["group"] for r in rows]
    gkf = GroupKFold(n_splits=n_splits)
    preds = {k: [None] * len(rows) for k in ("clf_oracle", "clf_heuristic", "clf_oracle_core6")}
    rpred = {k: [None] * len(rows) for k in preds}
    probs = [None] * len(rows)
    inf_ms = []
    for tr, te in gkf.split(np.zeros(len(rows)), groups=groups):
        trr = [rows[i] for i in tr]
        ter = [rows[i] for i in te]
        for name, labs, feats in (("clf_oracle", (y_or, r_or), ALL_FEATURES),
                                  ("clf_heuristic", (y_he, r_he), ALL_FEATURES),
                                  ("clf_oracle_core6", (y_or, r_or), CORE_FEATURES)):
            m = LaneClassifier(cfg, feats).fit(trr, [labs[0][i] for i in tr], [labs[1][i] for i in tr], seed)
            t0 = time.perf_counter()
            ln, rt, P = m.predict_rows(ter)
            if name == "clf_oracle":
                inf_ms.append((time.perf_counter() - t0) * 1000 / max(1, len(ter)))
                for i, p in zip(te, P):
                    probs[i] = p
            for i, l, r in zip(te, ln, rt):
                preds[name][i], rpred[name][i] = l, r
    preds["rules_only"] = [rules_only_route(r, cfg)[0] for r in rows]
    rpred["rules_only"] = [rules_only_route(r, cfg)[1] for r in rows]
    hl = [heuristic_label(r["seg"], cfg) for r in rows]
    preds["heuristic_labeller"] = [d.lane for d in hl]
    rpred["heuristic_labeller"] = [d.ratio for d in hl]
    # rules always win at inference: apply rule overrides to every router for the downstream view
    out: Dict[str, Any] = {"n_rows": len(rows), "n_groups": len(set(groups)), "routers": {},
                           "label_dist": {l: y_or.count(l) for l in LANES}}
    for name, p in preds.items():
        final = []
        for r, l in zip(rows, p):
            if r["rule"]:
                l = rules_only_route(r, cfg)[0]
            elif l == "externalize" and r["forbid"]:
                l = "compress"
            final.append(l)
        pr, rc, f1, sup = precision_recall_fscore_support(y_or, final, labels=LANES, zero_division=0)
        comp_idx = [i for i, (t, q) in enumerate(zip(y_or, final)) if t == "compress" and q == "compress"]
        mae = float(np.mean([abs(rpred[name][i] - r_or[i]) for i in comp_idx])) if comp_idx else None
        danger = sum(1 for r, q in zip(rows, final) if r["needed"] and q == "externalize")
        out["routers"][name] = {
            "per_lane": {l: {"precision": round(float(pr[i]), 3), "recall": round(float(rc[i]), 3),
                             "f1": round(float(f1[i]), 3), "support": int(sup[i])} for i, l in enumerate(LANES)},
            "macro_f1": round(float(f1_score(y_or, final, labels=LANES, average="macro", zero_division=0)), 3),
            "confusion": confusion_matrix(y_or, final, labels=LANES).tolist(),
            "ratio_mae": None if mae is None else round(mae, 3),
            "dangerous_errors": danger, "n_needed": sum(r["needed"] for r in rows),
            "externalize_rate": round(final.count("externalize") / len(final), 3),
        }
    # calibration of the oracle-trained classifier (held-out folds)
    P = np.array(probs)
    conf = P.max(1)
    pred = [LANES[i] for i in P.argmax(1)]
    bins = np.linspace(0.3, 1.0, 8)
    calib = []
    for a, b in zip(bins[:-1], bins[1:]):
        idx = [i for i, c in enumerate(conf) if a <= c < b or (b == 1.0 and c == 1.0)]
        if idx:
            calib.append({"bin": f"{a:.2f}-{b:.2f}", "n": len(idx),
                          "accuracy": round(float(np.mean([pred[i] == y_or[i] for i in idx])), 3),
                          "mean_conf": round(float(conf[idx].mean()), 3)})
    out["calibration"] = calib
    out["below_conf_threshold"] = int((conf < cfg.classifier_conf_threshold).sum())
    out["inference_ms_per_segment"] = round(float(np.mean(inf_ms)), 4)
    full = LaneClassifier(cfg, ALL_FEATURES).fit(rows, y_or, r_or, seed)
    out["feature_importances"] = sorted(((f, round(float(v), 4)) for f, v in
                                         zip(ALL_FEATURES, full.lane_model.feature_importances_)), key=lambda t: -t[1])
    out["note"] = ("Ground truth = oracle labels from workload gold. Heuristic-label training accuracy is circular "
                   "and is not reported as a result. All numbers are on held-out conversations (GroupKFold).")
    return out
