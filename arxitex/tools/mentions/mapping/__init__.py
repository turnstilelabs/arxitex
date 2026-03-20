"""Shared reference-to-artifact mapping utilities (single-file mapper)."""

from .ref_artifact_mapper import (
    GoldLinksBuildResult,
    MappingMatch,
    MappingResolution,
    MappingStats,
    Policy,
    TargetRegistry,
    VersionResolution,
    build_gold_links,
    build_target_registry,
    map_explicit_refs_to_artifacts,
    resolve_target_version,
)

__all__ = [
    "MappingMatch",
    "MappingResolution",
    "MappingStats",
    "Policy",
    "GoldLinksBuildResult",
    "TargetRegistry",
    "VersionResolution",
    "build_gold_links",
    "build_target_registry",
    "map_explicit_refs_to_artifacts",
    "resolve_target_version",
]
