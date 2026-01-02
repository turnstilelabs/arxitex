from __future__ import annotations

import math
from typing import Tuple

from arxitex.extractor.dependency_inference.dependency_mode import (
    DependencyInferenceConfig,
    DependencyInferenceMode,
)
from arxitex.extractor.models import ArtifactNode


def estimate_tokens_for_global(
    nodes: list[ArtifactNode], cfg: DependencyInferenceConfig
) -> int:
    """Very rough token estimate for including all artifacts in one prompt.

    We use a conservative heuristic 1 token ~= 4 chars.
    """

    total_chars = 0
    for n in nodes:
        total_chars += len(n.content or "")
        if cfg.global_include_proofs:
            pr = n.proof or ""
            total_chars += min(len(pr), cfg.global_proof_char_budget)
    return int(math.ceil(total_chars / 4))


def choose_mode_auto(
    nodes: list[ArtifactNode], cfg: DependencyInferenceConfig
) -> Tuple[DependencyInferenceMode, str, int]:
    """Choose dependency inference mode and provide reason + token estimate."""

    n = len(nodes)
    tok_est = estimate_tokens_for_global(nodes, cfg)

    # If the doc fits comfortably, we can use global reasoning.
    if n <= cfg.auto_max_nodes_global and tok_est <= cfg.auto_max_tokens_global:
        # Heuristic: for very small docs, global is cheap and usually adequate.
        if n <= 15:
            return (
                "global",
                f"auto: N={n} <= 15 and tok_est≈{tok_est} <= {cfg.auto_max_tokens_global}",
                tok_est,
            )
        return (
            "hybrid",
            f"auto: N={n} <= {cfg.auto_max_nodes_global} and tok_est≈{tok_est} <= {cfg.auto_max_tokens_global}",
            tok_est,
        )

    return (
        "pairwise",
        f"auto: fallback (N={n}, tok_est≈{tok_est}) exceeds thresholds (N<={cfg.auto_max_nodes_global}, tok<={cfg.auto_max_tokens_global})",
        tok_est,
    )
