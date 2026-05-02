from __future__ import annotations


def test_stage5_5_entry_uses_v9_scheduler_builder(monkeypatch):
    import tools.train_minimal_streetforward_stage5_5_multi_scene_v9 as entry

    class _DummyTrainer:
        pass

    called = {"base_v4_main": False}

    monkeypatch.setattr(entry, "_select_stage5_5_trainer", lambda _path: _DummyTrainer)

    def _fake_base_v4_main() -> None:
        called["base_v4_main"] = True
        assert entry.base.base.build_train_scheduler_from_cfg is entry.build_train_scheduler_v9_from_cfg
        assert entry.base.base.TRAINER_CLASS is _DummyTrainer

    def _forbidden_base_v8_main() -> None:
        raise AssertionError("stage5_5 entry must not call v8 main() because it rewires scheduler to v8")

    monkeypatch.setattr(entry.base.base, "main", _fake_base_v4_main)
    monkeypatch.setattr(entry.base, "main", _forbidden_base_v8_main)
    monkeypatch.setattr(
        entry.sys,
        "argv",
        ["train_minimal_streetforward_stage5_5_multi_scene_v9.py", "--config_file", "configs/minimal_streetforward_stage5_5_multi_scene_v9.yaml"],
    )

    entry.main()
    assert called["base_v4_main"] is True
