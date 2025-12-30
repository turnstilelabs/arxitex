from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DependencyInferenceMode = Literal["pairwise", "global", "hybrid", "auto"]


@dataclass(frozen=True)
class DependencyInferenceConfig:
    """Configuration for dependency inference strategies.

    Notes:
        - "global" mode uses statements + truncated proofs (Option B) to keep
          the prompt size bounded.
        - "hybrid" mode uses a global proposer to propose candidate edges, then
          runs the existing pairwise verifier on those candidates.
    """

    # --- auto mode thresholds ---
    auto_max_nodes_global: int = 30
    auto_max_tokens_global: int = 12_000

    # --- global prompt content controls ---
    # Option B: include proofs but truncated.
    global_include_proofs: bool = True
    global_proof_char_budget: int = 1200

    # --- hybrid caps (critical to avoid expensive redundancy) ---
    hybrid_topk_per_source: int = 8
    hybrid_max_total_candidates: int = 250

    # --- verification controls ---
    # Keep current behavior: pairwise verification includes proofs.
    hybrid_verify_with_proofs: bool = True
