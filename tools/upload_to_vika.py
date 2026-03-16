"""
Helper for uploading Minimal StreetForward experiment summaries to Vika using vika.py.

Usage (from a training script):

    from tools.upload_to_vika import upload_experiment_summary

    summary = {...}  # dict matching Vika fields
    upload_experiment_summary(cfg.log_dir, summary)

Environment variables:
- VIKA_TOKEN: required API token
- VIKA_DATASHEET_ID: target datasheet id (or URL) in Vika
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

try:
    from vika import Vika
except ImportError:  # pragma: no cover - optional dependency
    Vika = None  # type: ignore


def _load_default_summary(log_dir: str) -> Dict[str, Any]:
    """Load metrics_final.json from log_dir as a fallback summary."""
    path = os.path.join(log_dir, "metrics_final.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"metrics_final.json not found in log_dir={log_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_vika_config() -> Optional[Dict[str, Any]]:
    """Load Vika config from configs/vika_minimal_sf.yaml if present."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "configs", "vika_minimal_sf.yaml")
    if not os.path.isfile(cfg_path):
        return None
    cfg = OmegaConf.load(cfg_path)
    v = cfg.get("vika")
    if v is None:
        return None
    return OmegaConf.to_container(v, resolve=True)  # type: ignore[no-any-return]


def upload_experiment_summary(
    log_dir: str,
    summary_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Upload one experiment summary record to Vika.

    - log_dir: training run log directory (contains metrics_final.json).
    - summary_fields: dict to send as fields; if None, loads metrics_final.json.

    Fast-fail:
    - If vika.py is not installed, or env vars are missing, this function logs a warning and returns.
    - Any Vika API error is raised to the caller.
    """
    if Vika is None:
        logger.warning("vika.py is not installed; skip Vika upload.")
        return

    cfg = _load_vika_config() or {}
    if not cfg.get("enabled", True):
        logger.info("Vika upload disabled by configs/vika_minimal_sf.yaml; skip.")
        return

    token_env = cfg.get("token_env", "VIKA_TOKEN")
    token = os.environ.get(token_env)
    datasheet_id = cfg.get("datasheet_id") or os.environ.get("VIKA_DATASHEET_ID")
    if not token or not datasheet_id:
        logger.warning(
            "Vika config missing token or datasheet_id; skip upload for log_dir=%s.",
            log_dir,
        )
        return

    fields: Dict[str, Any]
    if summary_fields is not None:
        fields = dict(summary_fields)
    else:
        fields = _load_default_summary(log_dir)

    client = Vika(token)
    dst = client.datasheet(datasheet_id)

    logger.info("Uploading experiment summary to Vika datasheet=%s ...", datasheet_id)
    record = dst.records.create(fields)
    # record._id is the recordId
    logger.info("Vika upload succeeded. record_id=%s", getattr(record, "_id", "unknown"))

