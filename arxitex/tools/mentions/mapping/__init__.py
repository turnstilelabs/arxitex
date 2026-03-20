"""Shared reference-to-artifact mapping utilities (single-file mapper)."""

from .ref_artifact_mapper import (
    GoldLinksBuildResult,
    MappingStats,
    Policy,
    TargetRegistry,
    build_gold_links,
    build_target_registry,
    map_explicit_refs_to_artifacts,
    resolve_target_version,
)

__all__ = [
    "MappingStats",
    "Policy",
    "GoldLinksBuildResult",
    "TargetRegistry",
    "build_gold_links",
    "build_target_registry",
    "map_explicit_refs_to_artifacts",
    "resolve_target_version",
]
