from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Sequence

import numpy as np

from .index_format import stable_uint64
from .schema import EpisodePlanV3, RolloutPlanV3


DistributionName = Literal["repeat_refine", "shuffled_coverage", "high_block_repair"]

DISTRIBUTION_TYPE_IDS = {"repeat_refine": 1, "shuffled_coverage": 2, "high_block_repair": 3}
EPISODE_STAGE_IDS = {"prelude": 1, "repair_tail": 2}
TRAIN_2D_MODE_IDS = {"trainable": 1, "frozen_no_grad": 2, "auto": 3}
ORDER_TYPE_IDS = {
    "chronological": 1,
    "local": 2,
    "local_shuffle": 3,
    "stratified_shuffle": 4,
    "global_shuffle": 5,
}


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        out = node.get(key, default)
        return default if out is None else out
    if hasattr(node, key):
        out = getattr(node, key)
        return default if out is None else out
    return default


def _cfg_items(node: Any) -> list[tuple[Any, Any]]:
    if node is None or isinstance(node, (str, bytes, list, tuple)):
        return []
    if isinstance(node, dict) or hasattr(node, "items"):
        try:
            return list(node.items())
        except Exception:
            return []
    return []


def _int_weight_map(raw: Any, default: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, value in _cfg_items(raw):
        try:
            k = int(key)
            p = float(value)
        except Exception:
            continue
        if k > 0 and p > 0.0:
            out[int(k)] = float(p)
    return out or dict(default)


def _str_weight_map(raw: Any, default: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in _cfg_items(raw):
        try:
            p = float(value)
        except Exception:
            continue
        if p > 0.0:
            out[str(key)] = float(p)
    return out or dict(default)


def _weighted_choice(rng: Any, weights: dict[Any, float], *, default: Any) -> Any:
    items = [(k, float(v)) for k, v in dict(weights or {}).items() if float(v) > 0.0]
    total = sum(p for _, p in items)
    if not items or total <= 0.0:
        return default
    draw = rng.random() * total
    acc = 0.0
    for value, prob in items:
        acc += float(prob)
        if draw <= acc:
            return value
    return items[-1][0]


def _random_partition(rng: Any, total: int, parts: int) -> list[int]:
    total = int(max(1, total))
    parts = int(max(1, min(parts, total)))
    if parts == 1:
        return [int(total)]
    cuts = sorted(rng.sample(range(1, int(total)), int(parts) - 1))
    values = []
    prev = 0
    for cut in cuts + [int(total)]:
        values.append(int(cut - prev))
        prev = int(cut)
    return values


def _clamp_b_r_for_max_k(b: int, r: int, max_k: int, n: int, *, preserve_b: bool = False) -> tuple[int, int, str]:
    n = int(max(1, n))
    max_k = int(max(1, max_k))
    r = int(max(1, r))
    b = int(max(1, min(int(b), n)))
    if b * r <= max_k:
        return b, r, "none"
    if bool(preserve_b):
        b = int(max(1, min(b, max_k, n)))
        if b * r <= max_k:
            return b, r, "cap_b"
        r = int(max(1, max_k // max(1, b)))
        return b, r, "preserve_b_reduce_r"
    b = int(max(1, min(b, max_k // max(1, r))))
    if b * r <= max_k:
        return b, r, "reduce_b"
    r = int(max(1, max_k // max(1, b)))
    return b, r, "reduce_b_then_r"


@dataclass(frozen=True)
class RolloutDistributionSpec:
    name: str
    distribution_type: DistributionName
    phase: Literal["assimilation", "repair"]
    visit_kind: str
    b_choices: dict[int, float]
    r_choices: dict[int, float] = field(default_factory=dict)
    k_budget: dict[int, float] = field(default_factory=dict)
    order_weights: dict[str, float] = field(default_factory=dict)
    candidate_policy: str = "unvisited_preferred"
    train_2d_mode: Literal["trainable", "frozen_no_grad", "auto"] = "trainable"
    last_update_write: bool = True


@dataclass(frozen=True)
class CurriculumPhaseSpec:
    name: str
    start_step: int
    end_step: int
    sequence_target_frames: int
    min_frames: int
    allow_short: bool
    distribution_weights: dict[str, float]
    max_k_train_2d: dict[str, int]
    max_k_frozen_2d: dict[str, int]


@dataclass(frozen=True)
class EpisodeRecipeSpec:
    prelude_order_policy: str = "mixed_random"
    min_prelude_rollouts: int = 2
    max_prelude_rollouts: int = 8
    min_repair_rollouts: int = 0
    max_repair_rollouts: int = 4
    cover_target_ratio: float = 0.65
    repair_candidate_policy: str = "visited_preferred"
    train_2d_policy: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributionalRolloutSample:
    distribution_type: DistributionName
    episode_stage: Literal["prelude", "repair_tail"]
    order_type: str
    phase: str
    visit_kind: str
    positions: tuple[int, ...]
    repeat_budgets: tuple[int, ...]
    requested_b: int
    requested_k: int
    raw_b: int
    raw_r: int
    clamp_strategy: str
    max_k: int
    train_2d_mode: str
    candidate_pool: str
    visited_ratio_before: float
    visited_ratio_after: float
    repair_visited_ratio: float
    curriculum_phase_name: str
    curriculum_phase_id: int
    prelude_rollout_idx: int = -1
    repair_tail_idx: int = -1


def _default_distribution_specs(cfg: Any) -> dict[str, RolloutDistributionSpec]:
    raw = _cfg_get(cfg, "distributions", {}) or {}
    train_policy = dict(_cfg_get(_cfg_get(cfg, "episode_recipe", {}) or {}, "train_2d_policy", {}) or {})
    defaults = {
        "repeat_refine": {
            "phase": "assimilation",
            "visit_kind": "assimilation",
            "b_choices": {1: 0.55, 2: 0.45},
            "k_budget": {2: 0.10, 4: 0.30, 6: 0.30, 8: 0.20, 10: 0.10},
            "order": {"local": 0.7, "chronological": 0.3},
            "candidate_policy": "unvisited_preferred",
            "train_2d_mode": "trainable",
        },
        "shuffled_coverage": {
            "phase": "assimilation",
            "visit_kind": "assimilation",
            "b_choices": {3: 0.20, 4: 0.35, 6: 0.30, 8: 0.15},
            "r_choices": {1: 0.70, 2: 0.30},
            "order": {"local_shuffle": 0.35, "stratified_shuffle": 0.50, "global_shuffle": 0.15},
            "candidate_policy": "unvisited_preferred",
            "train_2d_mode": "trainable",
        },
        "high_block_repair": {
            "phase": "repair",
            "visit_kind": "repair",
            "b_choices": {6: 0.25, 8: 0.35, 10: 0.25, 12: 0.15},
            "r_choices": {1: 0.75, 2: 0.25},
            "order": {"global_shuffle": 1.0},
            "candidate_policy": "visited_preferred",
            "train_2d_mode": "frozen_no_grad",
            "last_update_write": False,
        },
    }
    out: dict[str, RolloutDistributionSpec] = {}
    for name, default in defaults.items():
        item = _cfg_get(raw, name, {}) or {}
        if item is not None and bool(_cfg_get(item, "enable", True)) is False:
            continue
        b_default = dict(default["b_choices"])
        r_default = dict(default.get("r_choices", {}))
        k_default = dict(default.get("k_budget", {}))
        out[name] = RolloutDistributionSpec(
            name=str(name),
            distribution_type=name,  # type: ignore[arg-type]
            phase=str(_cfg_get(item, "phase", default["phase"])),  # type: ignore[arg-type]
            visit_kind=str(_cfg_get(item, "visit_kind", default["visit_kind"])),
            b_choices=_int_weight_map(_cfg_get(item, "b_choices", default["b_choices"]), b_default),
            r_choices=_int_weight_map(_cfg_get(item, "r_choices", default.get("r_choices", {})), r_default) if r_default else {},
            k_budget=_int_weight_map(_cfg_get(item, "k_budget", default.get("k_budget", {})), k_default) if k_default else {},
            order_weights=_str_weight_map(_cfg_get(item, "order", default["order"]), dict(default["order"])),
            candidate_policy=str(_cfg_get(item, "candidate_policy", default["candidate_policy"])),
            train_2d_mode=str(train_policy.get(name, _cfg_get(item, "train_2d_mode", default["train_2d_mode"]))),  # type: ignore[arg-type]
            last_update_write=bool(_cfg_get(item, "last_update_write", default.get("last_update_write", True))),
        )
    return out


def _default_curriculum(cfg: Any) -> tuple[CurriculumPhaseSpec, ...]:
    raw = list(_cfg_get(cfg, "curriculum", []) or [])
    if not raw:
        raw = [
            {
                "name": "warmup",
                "start_step": 0,
                "end_step": 5000,
                "sequence_target_frames": 10,
                "min_frames": 10,
                "allow_short": False,
                "weights": {"repeat_refine": 0.35, "shuffled_coverage": 0.55, "high_block_repair": 0.10},
                "max_k": {
                    "train_2d": {"repeat_refine": 8, "shuffled_coverage": 8, "high_block_repair": 8},
                    "frozen_2d": {"repeat_refine": 12, "shuffled_coverage": 12, "high_block_repair": 12},
                },
            },
            {
                "name": "main",
                "start_step": 5000,
                "end_step": 30000,
                "sequence_target_frames": 16,
                "min_frames": 10,
                "allow_short": True,
                "weights": {"repeat_refine": 0.20, "shuffled_coverage": 0.55, "high_block_repair": 0.25},
                "max_k": {
                    "train_2d": {"repeat_refine": 8, "shuffled_coverage": 10, "high_block_repair": 12},
                    "frozen_2d": {"repeat_refine": 12, "shuffled_coverage": 16, "high_block_repair": 16},
                },
            },
            {
                "name": "hardening",
                "start_step": 30000,
                "end_step": 60010,
                "sequence_target_frames": 24,
                "min_frames": 8,
                "allow_short": True,
                "weights": {"repeat_refine": 0.10, "shuffled_coverage": 0.40, "high_block_repair": 0.50},
                "max_k": {
                    "train_2d": {"repeat_refine": 8, "shuffled_coverage": 10, "high_block_repair": 12},
                    "frozen_2d": {"repeat_refine": 12, "shuffled_coverage": 16, "high_block_repair": 20},
                },
            },
        ]
    phases: list[CurriculumPhaseSpec] = []
    for item in raw:
        max_k = _cfg_get(item, "max_k", {}) or {}
        phases.append(
            CurriculumPhaseSpec(
                name=str(_cfg_get(item, "name", f"phase{len(phases)}")),
                start_step=int(_cfg_get(item, "start_step", 0)),
                end_step=int(_cfg_get(item, "end_step", 2**31 - 1)),
                sequence_target_frames=int(_cfg_get(item, "sequence_target_frames", 10)),
                min_frames=int(_cfg_get(item, "min_frames", 8)),
                allow_short=bool(_cfg_get(item, "allow_short", True)),
                distribution_weights={str(k): float(v) for k, v in _cfg_items(_cfg_get(item, "weights", {}) or {}) if float(v) > 0.0},
                max_k_train_2d={str(k): int(v) for k, v in _cfg_items(_cfg_get(max_k, "train_2d", {}) or {})},
                max_k_frozen_2d={str(k): int(v) for k, v in _cfg_items(_cfg_get(max_k, "frozen_2d", {}) or {})},
            )
        )
    phases.sort(key=lambda x: int(x.start_step))
    return tuple(phases)


def _default_recipe(cfg: Any) -> EpisodeRecipeSpec:
    raw = _cfg_get(cfg, "episode_recipe", {}) or {}
    prelude = _cfg_get(raw, "prelude", {}) or {}
    repair = _cfg_get(raw, "repair_tail", {}) or {}
    return EpisodeRecipeSpec(
        prelude_order_policy=str(_cfg_get(prelude, "order_policy", "mixed_random")),
        min_prelude_rollouts=int(_cfg_get(prelude, "min_rollouts", 2)),
        max_prelude_rollouts=int(_cfg_get(prelude, "max_rollouts", 8)),
        min_repair_rollouts=int(_cfg_get(repair, "min_rollouts", 0)),
        max_repair_rollouts=int(_cfg_get(repair, "max_rollouts", 4)),
        cover_target_ratio=float(_cfg_get(prelude, "cover_target_ratio", 0.65)),
        repair_candidate_policy=str(_cfg_get(repair, "candidate_policy", "visited_preferred")),
        train_2d_policy={str(k): str(v) for k, v in _cfg_items(_cfg_get(raw, "train_2d_policy", {}) or {})},
    )


class DistributionalEpisodeCompiler:
    def __init__(self, scheduler: Any, cfg: Any) -> None:
        self.scheduler = scheduler
        self.cfg = cfg or {}
        self.distributions = _default_distribution_specs(cfg)
        self.curriculum = _default_curriculum(cfg)
        self.recipe = _default_recipe(cfg)
        if "repeat_refine" in self.distributions and max(self.distributions["repeat_refine"].b_choices) > 2:
            raise ValueError("scheduler_stage3_2.repeat_refine requires B <= 2")
        prelude_names = [name for name in ("repeat_refine", "shuffled_coverage") if name in self.distributions]
        if not prelude_names:
            raise ValueError("scheduler_stage3_2 requires repeat_refine and/or shuffled_coverage prelude distributions")
        if int(self.recipe.max_prelude_rollouts) < int(self.recipe.min_prelude_rollouts):
            raise ValueError("scheduler_stage3_2 prelude max_rollouts must be >= min_rollouts")
        if int(self.recipe.max_prelude_rollouts) < 1:
            raise ValueError("scheduler_stage3_2 requires at least one prelude rollout")
        if int(self.recipe.max_repair_rollouts) < int(self.recipe.min_repair_rollouts):
            raise ValueError("scheduler_stage3_2 repair_tail max_rollouts must be >= min_rollouts")
        for phase in self.curriculum:
            if int(phase.end_step) <= int(phase.start_step):
                raise ValueError(f"scheduler_stage3_2 curriculum phase {phase.name!r} requires end_step > start_step")

    def phase_for_step(self, step: int) -> CurriculumPhaseSpec:
        active = self.curriculum[0]
        for phase in self.curriculum:
            if int(step) >= int(phase.start_step):
                active = phase
            if int(phase.start_step) <= int(step) < int(phase.end_step):
                return phase
        return active

    def build_episode(self, *, step: int) -> EpisodePlanV3:
        phase = self.phase_for_step(int(step))
        segment_row, rows, _, sequence_frame_rule = self._sample_sequence_rows_for_phase(phase)
        scene_id, segment_id = self.scheduler._segment_ids(segment_row)
        sequence_id = int(
            stable_uint64(
                (
                    scene_id,
                    segment_id,
                    tuple(int(x["frame_idx"]) for x in rows),
                    self.scheduler.rng.getrandbits(63),
                )
            )
            & 0x7FFFFFFFFFFFFFFF
        )
        samples = self._sample_episode_samples(rows=rows, phase=phase)
        repair_positions_flat = [
            int(pos)
            for sample in samples
            if str(sample.episode_stage) == "repair_tail"
            for pos in tuple(sample.positions)
        ]
        repair_enabled = bool(repair_positions_flat)
        repair_hash = (
            int(stable_uint64((scene_id, segment_id, sequence_id, *repair_positions_flat, self.scheduler.rng.getrandbits(63))) & 0x7FFFFFFFFFFFFFFF)
            if repair_enabled
            else -1
        )
        rollouts: list[RolloutPlanV3] = []
        visit_counts: Dict[int, int] = {}
        last_visit_step_by_pos: Dict[int, int] = {}
        last_visit_context: Dict[str, Any] = {}
        step_offset = 0
        for ridx, sample in enumerate(samples):
            positions = [int(x) for x in sample.positions]
            history = [
                int(p)
                for p in range(int(rows.shape[0]))
                if int(visit_counts.get(int(p), 0)) > 0 and int(p) not in set(positions)
            ]
            if sample.phase == "repair" and not history:
                history = [int(p) for p in range(int(rows.shape[0])) if int(p) not in set(positions)]
            plan = self.scheduler._rollout_from_positions(
                rows=rows,
                scene_id=int(scene_id),
                segment_id=int(segment_id),
                sequence_id=int(sequence_id),
                positions=positions,
                repeat_budgets=[int(x) for x in sample.repeat_budgets],
                rollout_idx=int(ridx),
                rollouts_per_episode=int(len(samples)),
                phase=str(sample.phase),
                visit_kind=str(sample.visit_kind),
                history_positions=history,
                repair_positions=repair_positions_flat,
                repair_enabled=repair_enabled,
                repair_hash=int(repair_hash),
                episode_step_offset=int(step_offset),
                visit_counts=visit_counts,
                last_visit_step_by_pos=last_visit_step_by_pos,
                is_last_rollout=bool(ridx == len(samples) - 1),
                last_visit_context=last_visit_context,
                repair_round_idx=int(sample.repair_tail_idx),
                repair_pattern_name=f"B{len(positions)}R{int(round(float(sample.requested_k) / max(1, len(positions))))}"
                if sample.phase == "repair"
                else "",
                phase_max_inner_k=int(sample.max_k),
                requested_inner_k=int(sample.requested_k),
                requested_blocks_per_rollout=int(sample.requested_b),
                sequence_target_frames=int(sequence_frame_rule["target_frames"]),
                sequence_min_frames=int(sequence_frame_rule["min_frames"]),
                sequence_allow_short=bool(sequence_frame_rule["allow_short"]),
            )
            plan = self._attach_stage3_2_metadata(plan, sample, samples)
            rollouts.append(plan)
            step_offset += int(len(plan.steps))
        return EpisodePlanV3(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            episode_id=int(self.scheduler._episode_id_next),
            sequence_id=int(sequence_id),
            frame_set=tuple(int(row["frame_idx"]) for row in rows),
            keyframe_set=tuple(int(row["keyframe_idx"]) for row in rows),
            sampled_order=tuple(int(pos) for sample in samples for pos in tuple(sample.positions)),
            rollouts=tuple(rollouts),
            repair_enabled=bool(repair_enabled),
            metadata={
                "segment_row": int(segment_row),
                "scheduler_version": "stage3_2_distributional_episode_v1",
                "curriculum_phase_name": str(phase.name),
            },
        )

    def _sample_sequence_rows_for_phase(self, phase: CurriculumPhaseSpec) -> tuple[int, np.ndarray, tuple[int, ...], dict[str, Any]]:
        old = dict(self.scheduler.sequence_cfg)
        try:
            sequence = dict(old)
            sequence["min_frames"] = int(phase.min_frames)
            sequence["max_frames"] = int(phase.sequence_target_frames)
            sequence["frame_count_schedule"] = [
                {
                    "start_step": 0,
                    "target_frames": int(phase.sequence_target_frames),
                    "min_frames": int(phase.min_frames),
                    "allow_short": bool(phase.allow_short),
                }
            ]
            self.scheduler.sequence_cfg = sequence
            return self.scheduler._sample_sequence_rows()
        finally:
            self.scheduler.sequence_cfg = old

    def _max_k_for(self, phase: CurriculumPhaseSpec, dist: RolloutDistributionSpec) -> int:
        mode = str(dist.train_2d_mode)
        table = phase.max_k_frozen_2d if mode == "frozen_no_grad" else phase.max_k_train_2d
        fallback = 16 if mode == "frozen_no_grad" else 8
        return int(table.get(str(dist.distribution_type), fallback))

    def _sample_episode_samples(self, *, rows: np.ndarray, phase: CurriculumPhaseSpec) -> list[DistributionalRolloutSample]:
        prelude_names = [name for name in ("repeat_refine", "shuffled_coverage") if name in self.distributions]
        repair_enabled = "high_block_repair" in self.distributions and float(phase.distribution_weights.get("high_block_repair", 0.0)) > 0.0
        max_prelude = max(0, int(self.recipe.max_prelude_rollouts))
        min_prelude = min(max_prelude, max(0, int(self.recipe.min_prelude_rollouts)))
        prelude_count = self.scheduler.rng.randint(min_prelude, max_prelude) if max_prelude > min_prelude else max_prelude
        if len(prelude_names) >= 2 and max_prelude >= 2:
            prelude_count = max(2, int(prelude_count))
        prelude_order: list[str] = []
        if prelude_count > 0 and len(prelude_names) >= 2 and max_prelude >= 2:
            prelude_order.extend(["repeat_refine", "shuffled_coverage"])
        while len(prelude_order) < int(prelude_count):
            weights = {name: float(phase.distribution_weights.get(name, 1.0)) for name in prelude_names}
            prelude_order.append(str(_weighted_choice(self.scheduler.rng, weights, default=prelude_names[0])))
        if self.recipe.prelude_order_policy == "mixed_random":
            self.scheduler.rng.shuffle(prelude_order)

        repair_count = 0
        if repair_enabled:
            p = max(0.0, min(1.0, float(phase.distribution_weights.get("high_block_repair", 0.0))))
            repair_count = sum(1 for _ in range(max(0, int(self.recipe.max_repair_rollouts))) if self.scheduler.rng.random() < p)
            repair_count = max(int(self.recipe.min_repair_rollouts), min(int(self.recipe.max_repair_rollouts), int(repair_count)))

        samples: list[DistributionalRolloutSample] = []
        visited: set[int] = set()
        phase_id = list(self.curriculum).index(phase)
        for idx, name in enumerate(prelude_order):
            sample = self._sample_distribution(
                rows=rows,
                phase=phase,
                dist=self.distributions[str(name)],
                episode_stage="prelude",
                visited=visited,
                curriculum_phase_id=int(phase_id),
                prelude_rollout_idx=int(idx),
            )
            samples.append(sample)
            visited.update(int(x) for x in sample.positions)
        for idx in range(int(repair_count)):
            sample = self._sample_distribution(
                rows=rows,
                phase=phase,
                dist=self.distributions["high_block_repair"],
                episode_stage="repair_tail",
                visited=visited,
                curriculum_phase_id=int(phase_id),
                repair_tail_idx=int(idx),
            )
            samples.append(sample)
            visited.update(int(x) for x in sample.positions)
        return samples

    def _sample_distribution(
        self,
        *,
        rows: np.ndarray,
        phase: CurriculumPhaseSpec,
        dist: RolloutDistributionSpec,
        episode_stage: Literal["prelude", "repair_tail"],
        visited: set[int],
        curriculum_phase_id: int,
        prelude_rollout_idx: int = -1,
        repair_tail_idx: int = -1,
    ) -> DistributionalRolloutSample:
        n = int(rows.shape[0])
        max_k = self._max_k_for(phase, dist)
        visited_before = set(int(x) for x in visited)
        visited_ratio_before = float(len(visited_before)) / float(max(1, n))
        if dist.distribution_type == "repeat_refine":
            b = int(_weighted_choice(self.scheduler.rng, dist.b_choices, default=1))
            k = int(_weighted_choice(self.scheduler.rng, dist.k_budget, default=min(max_k, 4)))
            raw_b = int(b)
            raw_r = 0
            clamp_strategy = "k_budget"
            k = int(max(1, min(k, max_k)))
            b = int(max(1, min(2, b, k, n)))
            positions = self._choose_positions(n=n, count=b, visited=visited_before, policy=dist.candidate_policy)
            repeat_budgets = _random_partition(self.scheduler.rng, total=k, parts=len(positions))
            order_type = str(_weighted_choice(self.scheduler.rng, dist.order_weights, default="chronological"))
            positions = self._apply_order(positions, order_type=order_type, n=n)
        elif dist.distribution_type == "shuffled_coverage":
            b_raw = int(_weighted_choice(self.scheduler.rng, dist.b_choices, default=4))
            r_raw = int(_weighted_choice(self.scheduler.rng, dist.r_choices, default=1))
            raw_b = int(b_raw)
            raw_r = int(r_raw)
            b, r, clamp_strategy = _clamp_b_r_for_max_k(b_raw, r_raw, max_k, n)
            order_type = str(_weighted_choice(self.scheduler.rng, dist.order_weights, default="stratified_shuffle"))
            positions = self._choose_positions(n=n, count=b, visited=visited_before, policy=dist.candidate_policy, order_type=order_type)
            positions = self._apply_order(positions, order_type=order_type, n=n)
            repeat_budgets = [int(r) for _ in positions]
        else:
            b_raw = int(_weighted_choice(self.scheduler.rng, dist.b_choices, default=6))
            r_raw = int(_weighted_choice(self.scheduler.rng, dist.r_choices, default=1))
            raw_b = int(b_raw)
            raw_r = int(r_raw)
            b, r, clamp_strategy = _clamp_b_r_for_max_k(b_raw, r_raw, max_k, n, preserve_b=True)
            order_type = "global_shuffle"
            positions = self._choose_positions(n=n, count=b, visited=visited_before, policy="visited_preferred")
            positions = self._apply_order(positions, order_type=order_type, n=n)
            repeat_budgets = [int(r) for _ in positions]
        positions = [int(x) for x in positions[: max(1, min(len(positions), n))]]
        if not positions:
            positions = [0]
            repeat_budgets = [1]
        requested_k = int(sum(int(x) for x in repeat_budgets))
        visited_after = set(visited_before)
        visited_after.update(int(x) for x in positions)
        repair_visited_ratio = (
            float(sum(1 for pos in positions if int(pos) in visited_before)) / float(max(1, len(positions)))
            if episode_stage == "repair_tail"
            else 0.0
        )
        return DistributionalRolloutSample(
            distribution_type=dist.distribution_type,
            episode_stage=episode_stage,
            order_type=str(order_type),
            phase=str(dist.phase),
            visit_kind=str(dist.visit_kind),
            positions=tuple(int(x) for x in positions),
            repeat_budgets=tuple(int(x) for x in repeat_budgets[: len(positions)]),
            requested_b=int(len(positions)),
            requested_k=int(requested_k),
            raw_b=int(raw_b),
            raw_r=int(raw_r),
            clamp_strategy=str(clamp_strategy),
            max_k=int(max_k),
            train_2d_mode=str(dist.train_2d_mode),
            candidate_pool=str(dist.candidate_policy if episode_stage != "repair_tail" else "visited_preferred"),
            visited_ratio_before=float(visited_ratio_before),
            visited_ratio_after=float(len(visited_after)) / float(max(1, n)),
            repair_visited_ratio=float(repair_visited_ratio),
            curriculum_phase_name=str(phase.name),
            curriculum_phase_id=int(curriculum_phase_id),
            prelude_rollout_idx=int(prelude_rollout_idx),
            repair_tail_idx=int(repair_tail_idx),
        )

    def _choose_positions(
        self,
        *,
        n: int,
        count: int,
        visited: set[int],
        policy: str,
        order_type: str = "",
    ) -> list[int]:
        count = int(max(1, min(count, n)))
        unvisited = [p for p in range(int(n)) if int(p) not in visited]
        visited_l = [p for p in range(int(n)) if int(p) in visited]
        if order_type == "stratified_shuffle" and unvisited:
            return self._stratified_positions(n=n, count=count, primary=unvisited, fallback=visited_l)
        if str(policy) == "visited_preferred":
            primary, fallback = visited_l, unvisited
        else:
            primary, fallback = unvisited, visited_l
        self.scheduler.rng.shuffle(primary)
        self.scheduler.rng.shuffle(fallback)
        return [int(x) for x in (primary + fallback)[:count]]

    def _stratified_positions(self, *, n: int, count: int, primary: Sequence[int], fallback: Sequence[int]) -> list[int]:
        buckets: list[list[int]] = [[] for _ in range(int(count))]
        allowed = set(int(x) for x in list(primary))
        for pos in range(int(n)):
            if int(pos) not in allowed:
                continue
            bucket_idx = min(int(count) - 1, int(pos * int(count) / max(1, int(n))))
            buckets[int(bucket_idx)].append(int(pos))
        out: list[int] = []
        for bucket in buckets:
            if bucket:
                out.append(int(self.scheduler.rng.choice(bucket)))
        rest = [int(x) for x in list(primary) + list(fallback) if int(x) not in set(out)]
        self.scheduler.rng.shuffle(rest)
        out.extend(rest[: max(0, int(count) - len(out))])
        return out[: int(count)]

    def _apply_order(self, positions: Sequence[int], *, order_type: str, n: int) -> list[int]:
        out = [int(x) for x in positions]
        if str(order_type) == "chronological":
            return sorted(out)
        if str(order_type) in {"local", "local_shuffle"}:
            out = sorted(out)
            idx = 0
            while idx < len(out) - 1:
                if self.scheduler.rng.random() < 0.5:
                    out[idx], out[idx + 1] = out[idx + 1], out[idx]
                    idx += 2
                else:
                    idx += 1
            return out
        self.scheduler.rng.shuffle(out)
        return out

    def _attach_stage3_2_metadata(
        self,
        plan: RolloutPlanV3,
        sample: DistributionalRolloutSample,
        samples: Sequence[DistributionalRolloutSample],
    ) -> RolloutPlanV3:
        repeat_count = sum(1 for x in samples if x.distribution_type == "repeat_refine")
        shuffle_count = sum(1 for x in samples if x.distribution_type == "shuffled_coverage")
        repair_count = sum(1 for x in samples if x.distribution_type == "high_block_repair")
        repeat_k = sum(int(x.requested_k) for x in samples if x.distribution_type == "repeat_refine")
        shuffle_k = sum(int(x.requested_k) for x in samples if x.distribution_type == "shuffled_coverage")
        repair_k = sum(int(x.requested_k) for x in samples if x.distribution_type == "high_block_repair")
        phase = self.curriculum[int(sample.curriculum_phase_id)]
        meta = {
            "enabled": True,
            "distribution_type": str(sample.distribution_type),
            "distribution_type_id": int(DISTRIBUTION_TYPE_IDS.get(str(sample.distribution_type), 0)),
            "episode_stage": str(sample.episode_stage),
            "episode_stage_id": int(EPISODE_STAGE_IDS.get(str(sample.episode_stage), 0)),
            "order_type": str(sample.order_type),
            "order_type_id": int(ORDER_TYPE_IDS.get(str(sample.order_type), 0)),
            "train_2d_mode": str(sample.train_2d_mode),
            "train_2d_mode_id": int(TRAIN_2D_MODE_IDS.get(str(sample.train_2d_mode), 0)),
            "raw_B": int(sample.raw_b),
            "raw_R": int(sample.raw_r),
            "B": int(sample.requested_b),
            "R": int(round(float(sample.requested_k) / float(max(1, sample.requested_b)))),
            "R_mean": float(sample.requested_k) / float(max(1, sample.requested_b)),
            "K": int(sample.requested_k),
            "maxK": int(sample.max_k),
            "clamp_strategy": str(sample.clamp_strategy),
            "visited_ratio_before": float(sample.visited_ratio_before),
            "visited_ratio_after": float(sample.visited_ratio_after),
            "repair_visited_ratio": float(sample.repair_visited_ratio),
            "candidate_pool": str(sample.candidate_pool),
            "curriculum_phase_name": str(sample.curriculum_phase_name),
            "curriculum_phase_id": int(sample.curriculum_phase_id),
            "prelude_rollout_idx": int(sample.prelude_rollout_idx),
            "repair_tail_idx": int(sample.repair_tail_idx),
            "prelude_repeat_count": int(repeat_count),
            "prelude_shuffle_count": int(shuffle_count),
            "repair_tail_count": int(repair_count),
            "episode_distribution_rollout_count_repeat_refine": int(repeat_count),
            "episode_distribution_rollout_count_shuffled_coverage": int(shuffle_count),
            "episode_distribution_rollout_count_high_block_repair": int(repair_count),
            "episode_distribution_k_count_repeat_refine": int(repeat_k),
            "episode_distribution_k_count_shuffled_coverage": int(shuffle_k),
            "episode_distribution_k_count_high_block_repair": int(repair_k),
            "episode_distribution_weight_repeat_refine": float(phase.distribution_weights.get("repeat_refine", 0.0)),
            "episode_distribution_weight_shuffled_coverage": float(phase.distribution_weights.get("shuffled_coverage", 0.0)),
            "episode_distribution_weight_high_block_repair": float(phase.distribution_weights.get("high_block_repair", 0.0)),
        }
        request_meta = dict(getattr(plan, "request_meta", {}) or {})
        request_meta["iforward_stage3_2"] = meta
        stage23 = dict(request_meta.get("iforward_stage2_3", {}) or {})
        stage23["scheduler_version"] = "stage3_2_distributional_episode_v1"
        request_meta["iforward_stage2_3"] = stage23
        return dataclasses.replace(
            plan,
            shape_name=f"{meta['distribution_type']}_b{meta['B']}k{meta['K']}",
            rollout_phase=f"{meta['episode_stage']}_{meta['distribution_type']}",
            request_meta=request_meta,
        )


__all__ = [
    "DISTRIBUTION_TYPE_IDS",
    "EPISODE_STAGE_IDS",
    "ORDER_TYPE_IDS",
    "TRAIN_2D_MODE_IDS",
    "CurriculumPhaseSpec",
    "DistributionalEpisodeCompiler",
    "DistributionalRolloutSample",
    "EpisodeRecipeSpec",
    "RolloutDistributionSpec",
]
