from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    EVIDENCE = "evidence"
    BLOCK_LOSS = "block_loss"
    NEARBY_LOSS = "nearby_loss"

    PREFIX_LOSS = "prefix_loss"
    QUERY_LABEL = "query_label"
    AUX_LOSS = "aux_loss"

