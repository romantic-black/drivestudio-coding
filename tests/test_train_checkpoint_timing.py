from tools.train_minimal_streetforward_stage4_3_multi_scene_v4 import _periodic_checkpoint_due


def test_periodic_checkpoint_is_due_after_completed_step_count():
    assert not _periodic_checkpoint_due(0, 10000)
    assert not _periodic_checkpoint_due(9998, 10000)
    assert _periodic_checkpoint_due(9999, 10000)
    assert _periodic_checkpoint_due(19999, 10000)


def test_periodic_checkpoint_disabled_or_invalid_steps_are_not_due():
    assert not _periodic_checkpoint_due(-1, 10000)
    assert not _periodic_checkpoint_due(9999, 0)
    assert not _periodic_checkpoint_due(9999, None)
