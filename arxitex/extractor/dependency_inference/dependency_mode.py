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

    # --- Global per-paper cap on LLM-verified dependency pairs ---
    # Applies uniformly to both hybrid and pairwise strategies.
    #
    # Each mode first uses its own heuristics (global proposer for hybrid,
    # term/definition footprints for pairwise) to generate a deduped list of
    # candidate (source, target) pairs. If the resulting list has more than
    # ``max_total_pairs`` entries for a given paper, we skip dependency
    # inference for that paper entirely (with a clear warning) to avoid
    # excessive LLM calls.
    max_total_pairs: int = 100

    # (Reserved for future extensions; currently all verification includes proofs.)
