from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    EVIDENCE = "evidence"
    BLOCK_LOSS = "block_loss"
    NEARBY_LOSS = "nearby_loss"

    PREFIX_LOSS = "prefix_loss"
    QUERY_LABEL = "query_label"
    AUX_LOSS = "aux_loss"


class LongRole(str, Enum):
    EVIDENCE = "evidence"
    FINAL_HISTORY_RECON = "final_history_recon"
    FINAL_HISTORY_NVS = "final_history_nvs"
    FINAL_CURRENT_RECON = "final_current_recon"
    FINAL_CURRENT_NVS = "final_current_nvs"
