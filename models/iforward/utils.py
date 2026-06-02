from __future__ import annotations

import copy
from typing import Any, Optional


def cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    if hasattr(node, key):
        value = getattr(node, key)
        return default if value is None else value
    return default


def cfg_set(node: Any, key: str, value: Any) -> None:
    if isinstance(node, dict):
        node[key] = value
        return
    try:
        node[key] = value
        return
    except Exception:
        pass
    setattr(node, key, value)


def cfg_ensure_child(node: Any, key: str) -> Any:
    child = cfg_get(node, key, None)
    if child is not None:
        return child
    child = {}
    cfg_set(node, key, child)
    return cfg_get(node, key, child)


def clone_config(config: Any) -> Any:
    return copy.deepcopy(config)


def optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    return int(value)
