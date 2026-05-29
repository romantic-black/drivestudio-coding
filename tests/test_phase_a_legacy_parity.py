from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from streetforward_core.legacy.stage6_facade import PhaseAEventBuilderAdapter, PosteriorUpdaterAdapter, Stage6LegacyFacade
from streetforward_core.protocols.rollout import PHASE_A_NAME
from streetforward_core.train.stage6_phase_a_trainer import Stage6PhaseAFacadeTrainer, Stage6PhaseATrainer


class _ForwardOutput:
    def __init__(self):
        self.value = {"loss": 1.0, "roles": "legacy"}

    def to_legacy_dict(self):
        return dict(self.value)


class _Recipe:
    def __call__(self, batch):
        return _ForwardOutput()

    def train(self, mode=True):
        self.training = bool(mode)


class _Runtime:
    def forward(self, batch):
        return {"legacy": True}

    def validate_v9_phase_a(self, *args, **kwargs):
        return {"validation_path": "legacy"}


def _trainer():
    trainer = Stage6PhaseAFacadeTrainer.__new__(Stage6PhaseAFacadeTrainer)
    trainer.stage6_phase = PHASE_A_NAME
    trainer.recipe = _Recipe()
    trainer.runtime = _Runtime()
    trainer.runner = None
    trainer.facade = None
    return trainer


def test_stage6_phase_a_facade_trainer_alias_and_inheritance_boundary():
    assert Stage6PhaseATrainer is Stage6PhaseAFacadeTrainer
    assert not issubclass(Stage6PhaseAFacadeTrainer, MinimalStreetForwardStage5_4)


def test_stage6_phase_a_forward_preserves_legacy_dict_contract():
    trainer = _trainer()
    recipe_out = trainer.forward_recipe({})
    assert isinstance(recipe_out, _ForwardOutput)
    assert trainer.forward({}) == {"loss": 1.0, "roles": "legacy"}
    assert trainer({}) == {"loss": 1.0, "roles": "legacy"}


def test_stage6_phase_a_validation_method_is_explicit_legacy_proxy():
    assert _trainer().validate_v9_phase_a()["validation_path"] == "legacy"


def test_phase_a_event_builder_and_posterior_adapters_are_split():
    class Runtime:
        def __init__(self):
            self.calls = []

        def _build_stage6_event_from_measurement(self, *, local_state, measurement):
            self.calls.append(("build", local_state, measurement))
            return {"event": measurement["value"]}

        def _apply_event_update(self, *, local_state, event, ctx_vsm=None):
            self.calls.append(("update", local_state, event, ctx_vsm))
            return "next_state", "delta", {"aux": 1}

    runtime = Runtime()
    facade = Stage6LegacyFacade(runtime)
    event = PhaseAEventBuilderAdapter(facade).build_event(local_state="state", measurement={"value": 7})
    next_state, delta, aux = PosteriorUpdaterAdapter(facade).apply_update(local_state="state", event=event)
    assert event == {"event": 7}
    assert (next_state, delta, aux) == ("next_state", "delta", {"aux": 1})
    assert runtime.calls == [
        ("build", "state", {"value": 7}),
        ("update", "state", {"event": 7}, None),
    ]
