"""Shared reference-to-artifact mapping utilities (single-file mapper)."""

from .ref_artifact_mapper import (
    GoldLinksBuildResult,
    MappingStats,
    TargetRegistry,
    build_gold_links,
    build_target_registry,
)

__all__ = [
    "MappingStats",
    "GoldLinksBuildResult",
    "TargetRegistry",
    "build_gold_links",
    "build_target_registry",
]
