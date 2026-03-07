# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Freeway — JAX-native SDF game implementation.

A chicken crosses 10 lanes of traffic from bottom to top. Collisions push the
chicken back (downward impulse); reaching the top scores one point.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Action space (3 actions, matching ALE):
    0 — NOOP
    1 — UP
    2 — DOWN
"""

from typing import ClassVar

import chex
import jax
import jax.numpy as jnp

from atarax.env._base.traversal import TraversalGame, TraversalState
from atarax.env.hud import render_score
from atarax.env.sdf import (
    finalise_rgb,
    make_canvas,
    paint_layer,
    paint_sdf,
    render_rect_pool,
    sdf_rect,
)
from atarax.game import AtaraxParams

# ── Geometry
_WORLD_W: float = 160.0
_WORLD_H: float = 210.0

_CHICKEN_X: float = 80.0
_CHICKEN_HW: float = 4.0
_CHICKEN_HH: float = 5.0
_CHICKEN_SPEED: float = 2.5

_SAFE_TOP_Y: float = 10.0  # crossing threshold
_SAFE_BOT_Y: float = 190.0  # reset threshold (bottom of field)

_ROAD_TOP_Y: float = 20.0
_ROAD_BOT_Y: float = 185.0
_NUM_LANES: int = 10
_LANE_H: float = 16.0  # (_ROAD_BOT_Y - _ROAD_TOP_Y) / _NUM_LANES ≈ 16.5; use 16

_CAR_HW: float = 7.0
_CAR_HH: float = 3.5
_CARS_PER_LANE: int = 2   # 2 cars per lane — matches ALE default density

# Time limit: 7200 emulated frames (= 1800 agent steps at 4× skip)
_TIME_LIMIT: int = 7200

# Lane signed speeds (positive = right, negative = left).
# ALE Freeway layout: top 5 lanes (index 0-4, low y) → LEFT; bottom 5 (index 5-9, high y) → RIGHT.
# Speeds vary significantly per lane — some slow trucks, some fast cars.
# Range ≈ 0.5–1.8 px/frame (slower than original, matching observed ALE pacing).
_LANE_SPEEDS_PY: tuple[float, ...] = (
    -0.6,  # lane 0 (topmost) — slow, left
    -1.4,  # lane 1 — moderate, left
    -0.8,  # lane 2 — slow-medium, left
    -1.8,  # lane 3 — fast, left
    -0.5,  # lane 4 — slow trucks, left
    #  ── centre divider at y=100 ──
    0.7,   # lane 5 — slow, right
    1.6,   # lane 6 — fast, right
    0.5,   # lane 7 — slow trucks, right
    1.2,   # lane 8 — moderate, right
    0.9,   # lane 9 (bottommost) — moderate, right
)
_LANE_SPEEDS = jnp.array(_LANE_SPEEDS_PY, dtype=jnp.float32)

# Centre divider y — solid stripe between top-half and bottom-half lanes
_DIVIDER_Y: float = _ROAD_TOP_Y + 5 * _LANE_H  # = 100.0


# Initial car x positions: 2 cars evenly spaced per lane.
# Top half (lanes 0-4) and bottom half (lanes 5-9) are offset by half a gap
# from each other so the two traffic flows look visually independent.
def _init_cars() -> jnp.ndarray:
    """Return initial (10, 2, 2) float32 obstacle array [x, active=1]."""
    xs = []
    gap = _WORLD_W / _CARS_PER_LANE  # 80 px between 2 cars
    for lane in range(_NUM_LANES):
        # Small per-lane stagger; bottom half (lanes 5-9) offset by gap/2
        base_offset = (lane % 5) * (gap / _NUM_LANES)
        half_offset = (gap * 0.5) if lane >= 5 else 0.0
        lane_xs = [(base_offset + half_offset + i * gap) % _WORLD_W
                   for i in range(_CARS_PER_LANE)]
        xs.append(lane_xs)
    x_arr = jnp.array(xs, dtype=jnp.float32)  # (10, 2)
    active = jnp.ones((_NUM_LANES, _CARS_PER_LANE), dtype=jnp.float32)
    return jnp.stack([x_arr, active], axis=-1)  # (10, 2, 2)


_INIT_CARS = _init_cars()

# Lane centre y positions — Python tuple for static render loops, JAX array for physics
_LANE_CYS_PY: tuple[float, ...] = tuple(
    _ROAD_TOP_Y + i * _LANE_H + _LANE_H * 0.5 for i in range(_NUM_LANES)
)
_LANE_CYS = jnp.array(_LANE_CYS_PY, dtype=jnp.float32)

# ── Colours
_COL_BG = jnp.array([0.15, 0.55, 0.12], dtype=jnp.float32)  # grass green
_COL_ROAD = jnp.array([0.25, 0.25, 0.25], dtype=jnp.float32)  # dark grey
_COL_LANE_LINE = jnp.array([0.65, 0.65, 0.45], dtype=jnp.float32)
_COL_CHICKEN = jnp.array([0.95, 0.95, 0.70], dtype=jnp.float32)  # cream
_COL_TIMER = jnp.array([0.95, 0.75, 0.10], dtype=jnp.float32)  # amber
_COL_SCORE = jnp.array([1.0, 0.902, 0.725], dtype=jnp.float32)

# Per-lane car colours matching the original ALE Freeway palette
_CAR_COLOURS: tuple[jnp.ndarray, ...] = (
    jnp.array([0.95, 0.85, 0.10], dtype=jnp.float32),  # lane 0 — yellow
    jnp.array([0.20, 0.50, 0.90], dtype=jnp.float32),  # lane 1 — blue
    jnp.array([0.90, 0.40, 0.10], dtype=jnp.float32),  # lane 2 — orange
    jnp.array([0.75, 0.75, 0.75], dtype=jnp.float32),  # lane 3 — silver
    jnp.array([0.90, 0.25, 0.65], dtype=jnp.float32),  # lane 4 — pink
    jnp.array([0.25, 0.85, 0.85], dtype=jnp.float32),  # lane 5 — cyan
    jnp.array([0.95, 0.85, 0.10], dtype=jnp.float32),  # lane 6 — yellow
    jnp.array([0.90, 0.40, 0.10], dtype=jnp.float32),  # lane 7 — orange
    jnp.array([0.90, 0.25, 0.65], dtype=jnp.float32),  # lane 8 — pink
    jnp.array([0.20, 0.50, 0.90], dtype=jnp.float32),  # lane 9 — blue
)


@chex.dataclass
class FreewayParams(AtaraxParams):
    """Static configuration for Freeway."""

    max_steps: int = 10000


@chex.dataclass
class FreewayState(TraversalState):
    """
    Freeway game state.

    Extends `TraversalState`.  Field mapping:
    - `player_x`      → stubbed (fixed at 80.0)
    - `player_y`      → `chicken_y`
    - `jump_vy`       → `pushback_vy` (downward impulse after collision)
    - `obstacles`     → cars `(10, 3, 2)` [x, active]
    - `obstacle_speed`→ `_LANE_SPEEDS` (constant, stored for base-class helpers)
    - `crossings`     → total successful crossings (= score)
    - `timer`         → `time_remaining` (counts down from 7200)

    No extra fields are needed; all Freeway-specific state is covered by the
    inherited `TraversalState` fields.
    """


class Freeway(TraversalGame):
    """
    Freeway implemented as a pure-JAX function suite.

    All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.
    """

    num_actions: ClassVar[int] = 3
    game_id: ClassVar[str] = "freeway"

    def _reset(self, rng: chex.PRNGKey) -> FreewayState:
        """Return the canonical initial game state."""
        return FreewayState(
            # TraversalState fields
            player_x=jnp.float32(_CHICKEN_X),  # fixed — never changes
            player_y=jnp.float32(_SAFE_BOT_Y),  # start at bottom
            jump_vy=jnp.float32(0.0),  # pushback_vy starts at 0
            obstacles=_INIT_CARS,
            obstacle_speed=_LANE_SPEEDS,
            crossings=jnp.int32(0),
            timer=jnp.int32(_TIME_LIMIT),
            # AtariState fields
            lives=jnp.int32(0),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=rng,
        )

    def _step_physics(
        self,
        state: FreewayState,
        action: chex.Array,
        params: FreewayParams,
        _rng: chex.PRNGKey,
    ) -> FreewayState:
        """Advance the game by one emulated frame (branch-free)."""

        # ── 1. Chicken movement ───────────────────────────────────────────────
        dy = jnp.where(
            action == jnp.int32(1),
            jnp.float32(-_CHICKEN_SPEED),
            jnp.where(
                action == jnp.int32(2), jnp.float32(_CHICKEN_SPEED), jnp.float32(0.0)
            ),
        )
        chicken_y = state.player_y + dy

        # ── 2. Pushback decay and apply ───────────────────────────────────────
        pushback_vy = state.jump_vy * jnp.float32(0.8)
        chicken_y = chicken_y + pushback_vy

        # ── 3. Move cars (wrap toroidally) ────────────────────────────────────
        cars = self._move_obstacles(
            state.obstacles,
            state.obstacle_speed,
            _WORLD_W,
            _CAR_HW,
        )

        # ── 4. Collision: chicken vs all cars ─────────────────────────────────
        # Flatten (10, 3) → (30,) for vectorised AABB
        car_xs = cars[:, :, 0].reshape(-1)  # (30,)
        car_active = cars[:, :, 1].reshape(-1)  # (30,)

        # Lane centre y for each car — tile lane indices (10,) × 3
        lane_ys = jnp.repeat(_LANE_CYS, _CARS_PER_LANE)  # (30,)

        hit_mask = self._player_obstacle_hit(
            jnp.float32(_CHICKEN_X),
            chicken_y,
            _CHICKEN_HW,
            _CHICKEN_HH,
            car_xs,
            lane_ys,
            car_active > jnp.float32(0.5),
            _CAR_HW,
            _CAR_HH,
        )
        any_hit = jnp.any(hit_mask)

        # On collision: add downward pushback impulse
        new_pushback = jnp.where(
            any_hit,
            pushback_vy + jnp.float32(8.0),
            pushback_vy,
        )

        # Clamp chicken to bottom boundary (can't go below start position)
        chicken_y = jnp.minimum(chicken_y, jnp.float32(_SAFE_BOT_Y))

        # ── 5. Successful crossing ────────────────────────────────────────────
        crossed = chicken_y - jnp.float32(_CHICKEN_HH) <= jnp.float32(_SAFE_TOP_Y)
        new_score = state.score + jnp.where(crossed, jnp.int32(1), jnp.int32(0))
        new_crossings = state.crossings + jnp.where(crossed, jnp.int32(1), jnp.int32(0))
        # Reset chicken to bottom on crossing
        chicken_y = jnp.where(crossed, jnp.float32(_SAFE_BOT_Y), chicken_y)
        # Reset pushback on crossing too
        new_pushback = jnp.where(crossed, jnp.float32(0.0), new_pushback)

        reward = jnp.where(crossed, jnp.float32(1.0), jnp.float32(0.0))

        # ── 6. Timer and done ─────────────────────────────────────────────────
        new_timer = state.timer - jnp.int32(1)
        done = new_timer <= jnp.int32(0)

        return state.__replace__(
            player_y=chicken_y,
            jump_vy=new_pushback,
            obstacles=cars,
            crossings=new_crossings,
            timer=new_timer,
            score=new_score,
            reward=state.reward + reward,
            done=done,
            step=state.step + jnp.int32(1),
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: FreewayState,
        action: chex.Array,
        params: FreewayParams,
    ) -> FreewayState:
        """Advance the game by one agent step (4 emulated frames)."""
        state = state.__replace__(reward=jnp.float32(0.0))

        def physics_step(i: int, s: FreewayState) -> FreewayState:
            return self._step_physics(s, action, params, jax.random.fold_in(rng, i))

        state = jax.lax.fori_loop(0, 4, physics_step, state)
        return state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: FreewayState) -> chex.Array:
        """
        Render the current game state as an RGB frame.

        Returns
        -------
        frame : chex.Array
            uint8[210, 160, 3] — RGB image.
        """
        canvas = make_canvas(_COL_BG)

        # Layer 1 — Road (single dark strip covering all 10 lanes)
        road_cy = (_ROAD_TOP_Y + _ROAD_BOT_Y) * 0.5
        road_hh = (_ROAD_BOT_Y - _ROAD_TOP_Y) * 0.5
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                jnp.float32(80.0),
                jnp.float32(road_cy),
                jnp.float32(80.0),
                jnp.float32(road_hh),
            ),
            _COL_ROAD,
        )

        # Layer 2 — Lane dividers (dashed lines between lanes, solid centre divider)
        for lane in range(1, _NUM_LANES):
            line_y = _ROAD_TOP_Y + lane * _LANE_H
            is_centre = (lane == 5)  # solid white divider between top/bottom halves
            if is_centre:
                canvas = paint_sdf(
                    canvas,
                    sdf_rect(
                        jnp.float32(80.0),
                        jnp.float32(line_y),
                        jnp.float32(80.0),
                        jnp.float32(1.5),
                    ),
                    _COL_LANE_LINE,
                )
            else:
                for seg in range(5):
                    seg_x = 20.0 + seg * 32.0
                    canvas = paint_sdf(
                        canvas,
                        sdf_rect(
                            jnp.float32(seg_x),
                            jnp.float32(line_y),
                            jnp.float32(7.0),
                            jnp.float32(0.5),
                        ),
                        _COL_LANE_LINE,
                    )

        # Layer 3 — Cars (render_rect_pool per lane, per-lane colour)
        for lane in range(_NUM_LANES):
            lane_y = _LANE_CYS_PY[lane]  # Python float — safe in static JIT loop
            car_xs_lane = state.obstacles[lane, :, 0]  # (3,)
            car_active_lane = state.obstacles[
                lane, :, 1
            ]  # (3,) float, 1=active 0=inactive
            car_ys_lane = jnp.full((_CARS_PER_LANE,), jnp.float32(lane_y))
            car_pool = jnp.stack(
                [car_xs_lane, car_ys_lane, car_active_lane], axis=-1
            )  # (3, 3)
            car_mask = render_rect_pool(car_pool, _CAR_HW, _CAR_HH)  # (210, 160)
            canvas = paint_layer(canvas, car_mask, _CAR_COLOURS[lane])

        # Layer 4 — Chicken
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                jnp.float32(_CHICKEN_X),
                state.player_y,
                jnp.float32(_CHICKEN_HW),
                jnp.float32(_CHICKEN_HH),
            ),
            _COL_CHICKEN,
        )

        # Layer 5 — Timer bar (top strip, width proportional to time remaining)
        timer_frac = jnp.clip(
            state.timer.astype(jnp.float32) / jnp.float32(_TIME_LIMIT),
            jnp.float32(0.0),
            jnp.float32(1.0),
        )
        bar_hw = timer_frac * jnp.float32(76.0)
        bar_cx = jnp.float32(80.0) - jnp.float32(76.0) + bar_hw
        canvas = paint_sdf(
            canvas,
            sdf_rect(bar_cx, jnp.float32(5.0), bar_hw, jnp.float32(3.0)),
            _COL_TIMER,
        )

        # Layer 6 — Score HUD
        canvas = render_score(canvas, state.score, colour=_COL_SCORE)

        return finalise_rgb(canvas)
