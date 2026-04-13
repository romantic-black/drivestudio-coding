from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AssetConfig:
    enable: bool
    root: str
    use_prebuilt_assets: bool
    missing_policy: str


def normalize_missing_policy(raw: Any) -> str:
    policy = str(raw).strip().lower()
    if policy not in {"error", "rebuild", "ignore"}:
        raise ValueError(
            f"assets.missing_policy must be one of ['error', 'rebuild', 'ignore'], got {raw!r}"
        )
    return policy


def require_manifest_fields(manifest: Dict[str, Any], *, asset_type: str) -> None:
    required = (
        "asset_type",
        "schema_version",
        "asset_id",
        "dataset",
        "scene_id",
        "source_data_fingerprint",
        "config_fingerprint",
        "implementation_fingerprint",
    )
    for k in required:
        if k not in manifest:
            raise ValueError(f"{asset_type} manifest missing required field: {k}")
    if str(manifest["asset_type"]) != asset_type:
        raise ValueError(
            f"manifest asset_type mismatch: expected {asset_type!r}, got {manifest['asset_type']!r}"
        )
    if int(manifest["schema_version"]) != int(SCHEMA_VERSION):
        raise ValueError(
            f"manifest schema_version mismatch: expected {SCHEMA_VERSION}, got {manifest['schema_version']}"
        )
    if asset_type == "streetforward_scene_asset":
        for k in ("num_frames", "num_cams", "image_table_version"):
            if k not in manifest:
                raise ValueError(f"{asset_type} manifest missing required field: {k}")
