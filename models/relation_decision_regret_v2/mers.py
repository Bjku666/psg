"""CPU MERS utility learner over frozen pair/predicate evidence."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from .predicate_substitution_regret import _native_pred
from models.relation_decision_regret_v1.legal_oracle import _predicate_options


def action_features(row: Mapping, predicate: int, native: int | None = None) -> np.ndarray:
    scores = np.asarray(row["pred_scores"], dtype=float)[1:]
    native = _native_pred(row) if native is None else int(native)
    pair = float(row.get("score", 0.0))
    hidden = np.asarray(row.get("pair_features", np.zeros(384)), dtype=float).reshape(-1)
    if hidden.size:
        # Stable low-dimensional moments avoid a 384x56 parameter explosion.
        h = np.asarray([hidden.mean(), hidden.std(), hidden[: hidden.size // 2].mean(),
                        hidden[hidden.size // 2 :].mean()])
    else:
        h = np.zeros(4)
    probs = np.exp(scores - np.max(scores)); probs /= max(float(probs.sum()), 1e-12)
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    top = np.argsort(-scores, kind="stable")
    margin = float(scores[top[0]] - scores[top[1]]) if len(top) > 1 else 0.0
    return np.concatenate(([pair, float(scores[predicate]), float(scores[predicate] - scores[native]),
                            float(predicate == native), margin, entropy], h))


@dataclass
class MERS:
    model: object
    depth: int | str = 3

    @classmethod
    def fit(cls, rows: Sequence[tuple[Mapping, int, float]], depth: int | str = 3,
            kind: str = "hist") -> "MERS":
        x = np.stack([action_features(row, predicate) for row, predicate, _ in rows])
        y = np.asarray([utility for _, _, utility in rows], dtype=float)
        if kind == "ridge":
            model = Ridge(alpha=1.0).fit(x, y)
        else:
            model = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05,
                                                   max_leaf_nodes=15, l2_regularization=1e-2,
                                                   random_state=0).fit(x, y)
        return cls(model, depth)

    def score(self, row: Mapping, predicate: int) -> float:
        return float(self.model.predict(action_features(row, predicate)[None])[0])

    def choose(self, row: Mapping) -> int:
        native = _native_pred(row)
        options = [native] + [int(p) for p in _predicate_options(row, self.depth) if int(p) != native]
        values = [(self.score(row, p), p) for p in options]
        # KEEP utility is exactly zero, independent of model calibration.
        best = max(values, key=lambda x: (x[0], -x[1]))
        return int(best[1]) if best[0] > 0.0 else int(native)

    def choose_many(self, rows: Sequence[Mapping]) -> list[int]:
        """Vectorized decoder used by the fit/dev runner."""
        choices = []
        feature_rows = []
        option_rows = []
        for row in rows:
            native = _native_pred(row)
            options = [native] + [int(p) for p in _predicate_options(row, self.depth) if int(p) != native]
            option_rows.append(options)
            feature_rows.extend(action_features(row, p) for p in options)
        if not feature_rows:
            return []
        scores = np.asarray(self.model.predict(np.stack(feature_rows)), dtype=float)
        offset = 0
        for options in option_rows:
            values = scores[offset:offset + len(options)]
            offset += len(options)
            best = int(np.argmax(values))
            choices.append(int(options[best]) if float(values[best]) > 0.0 else int(options[0]))
        return choices
