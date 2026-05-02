from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def test_stage6_0_entry_uses_v10_scheduler_builder(monkeypatch):
    import tools.train_minimal_streetforward_stage6_0_multi_scene_v10 as entry

    class _DummyTrainer:
        pass

    called = {"base_v4_main": False}

    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "cfg.yaml"
        cfg_path.write_text(
            "model:\n"
            "  stage: \"6_0\"\n"
            "scheduler_v10:\n"
            "  enable: true\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(entry, "_select_stage6_0_trainer", lambda _path: _DummyTrainer)

        def _fake_base_v4_main() -> None:
            called["base_v4_main"] = True
            assert entry.base.base.build_train_scheduler_from_cfg is entry.build_train_scheduler_v10_from_cfg
            assert entry.base.base.TRAINER_CLASS is _DummyTrainer

        monkeypatch.setattr(entry.base.base, "main", _fake_base_v4_main)
        monkeypatch.setattr(
            entry.sys,
            "argv",
            ["train_minimal_streetforward_stage6_0_multi_scene_v10.py", "--config_file", str(cfg_path)],
        )
        entry.main()

    assert called["base_v4_main"] is True


def test_stage6_0_select_trainer_rejects_wrong_scheduler(tmp_path):
    import tools.train_minimal_streetforward_stage6_0_multi_scene_v10 as entry

    p = tmp_path / "bad.yaml"
    p.write_text(
        "model:\n"
        "  stage: \"6_0\"\n"
        "scheduler_v10:\n"
        "  enable: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scheduler_v10.enable=true"):
        entry._select_stage6_0_trainer(str(p))

