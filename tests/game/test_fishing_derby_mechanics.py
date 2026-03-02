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

"""Physics-level mechanics tests for Fishing Derby.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_fishing_derby_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.game import AtaraxParams
from atarax.games.fishing_derby import (
    FishingDerby,
    _CATCH_DX,
    _CATCH_DY,
    _DOCK_SPEED,
    _FISH_X0,
    _FISH_Y0,
    _FISH_DRIFT,
    _FISH_VALUES,
    _LINE_SPEED,
    _MAX_STEPS,
    _PLAYER_DOCK_START,
    _SCORE_LIMIT,
    _SHARK_H,
    _SHARK_SPEED,
    _SHARK_W,
    _SHARK_Y,
    _WATER_Y0,
)

_KEY = jax.random.PRNGKey(0)
_GAME = FishingDerby()
_INIT = _GAME._reset(_KEY)


def _state(**kw):
    """Return a fresh initial state with optional field overrides."""
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Action effects — player dock x and line depth
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "action,exp_dx,exp_dly",
    [
        pytest.param(0, 0.0, 0.0, id="noop"),
        pytest.param(2, 0.0, -_LINE_SPEED, id="up"),
        pytest.param(5, 0.0, +_LINE_SPEED, id="down"),
        pytest.param(3, +_DOCK_SPEED, 0.0, id="right"),
        pytest.param(4, -_DOCK_SPEED, 0.0, id="left"),
        pytest.param(6, +_DOCK_SPEED, -_LINE_SPEED, id="upright_diagonal"),
    ],
)
def test_action_moves_player(action, exp_dx, exp_dly):
    # Start from mid-dock and mid-water so clamps don't suppress movement.
    state = _state(
        player_x=jnp.float32(30.0),
        player_line_y=jnp.float32(130.0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(action))
    actual_dx = float(new_state.player_x) - 30.0
    actual_dly = float(new_state.player_line_y) - 130.0
    assert abs(actual_dx - exp_dx) < 1e-4, (
        f"action {action}: player_x delta expected {exp_dx}, got {actual_dx}"
    )
    assert abs(actual_dly - exp_dly) < 1e-4, (
        f"action {action}: line_y delta expected {exp_dly}, got {actual_dly}"
    )


# ---------------------------------------------------------------------------
# Catch mechanics
# ---------------------------------------------------------------------------
def test_catching_shallow_fish_scores_2():
    """Player line at shallow fish position → +2 points."""
    # fish[0]: initial x=20, y=90, value=2; after one drift step x≈20.3
    state = _state(
        player_x=jnp.float32(20.0),
        player_line_y=jnp.float32(90.0),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.score) == 2, (
        f"Expected score 2 after catching shallow fish, got {int(new_state.score)}"
    )
    assert abs(float(new_state.reward) - 2.0) < 1e-4


def test_catching_deep_fish_scores_6():
    """Player line at deep fish position → +6 points."""
    # fish[4]: y=150, value=6; override x to player-reachable position
    fish_x = _INIT.fish_x.at[4].set(jnp.float32(40.0))
    state = _state(
        player_x=jnp.float32(40.0),
        player_line_y=jnp.float32(150.0),
        fish_x=fish_x,
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.score) == 6, (
        f"Expected score 6 after catching deep fish, got {int(new_state.score)}"
    )
    assert abs(float(new_state.reward) - 6.0) < 1e-4


def test_cpu_catch_gives_negative_reward():
    """When CPU catches a fish and player catches nothing, reward is negative."""
    # Place fish[4] near CPU dock (x≈130), CPU line at its depth.
    fish_x = jnp.array([20.0, 35.0, 50.0, 30.0, 130.0], dtype=jnp.float32)
    state = _state(
        # Player is at the surface far from any fish.
        player_x=jnp.float32(_PLAYER_DOCK_START),
        player_line_y=jnp.float32(_WATER_Y0),
        # CPU is right on top of fish[4].
        cpu_x=jnp.float32(130.0),
        cpu_line_y=jnp.float32(150.0),
        fish_x=fish_x,
        cpu_score=jnp.int32(0),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert float(new_state.reward) < 0.0, (
        f"Expected negative reward when CPU catches fish, got {float(new_state.reward)}"
    )
    assert int(new_state.cpu_score) > int(state.cpu_score)


# ---------------------------------------------------------------------------
# Shark mechanic
# ---------------------------------------------------------------------------
def test_shark_cuts_line_resets_line_y():
    """Shark AABB overlapping player line → line_y reset to water surface."""
    # Shark will move: new_shark_x = shark_x + shark_dx = 25.0 + 1.5 = 26.5
    # Player at x=30, line_y=134 → overlaps shark box [26.5, 42.5) × [130, 138)
    state = _state(
        player_x=jnp.float32(30.0),
        player_line_y=jnp.float32(134.0),
        shark_x=jnp.float32(25.0),
        shark_dx=jnp.float32(_SHARK_SPEED),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert abs(float(new_state.player_line_y) - float(_WATER_Y0)) < 1e-4, (
        f"Expected line_y reset to {_WATER_Y0}, got {float(new_state.player_line_y)}"
    )


# ---------------------------------------------------------------------------
# Episode end conditions
# ---------------------------------------------------------------------------
def test_episode_ends_when_player_scores_99():
    """Player score reaching ≥99 → done."""
    # score=97; catch shallow fish (2 pts) → score=99 ≥ 99 → done
    state = _state(
        player_x=jnp.float32(20.0),
        player_line_y=jnp.float32(90.0),
        score=jnp.int32(97),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert bool(new_state.done), (
        f"Expected done=True when score reaches {_SCORE_LIMIT}"
    )


def test_episode_ends_at_max_steps():
    """Episode ends (via _step) when episode_step reaches _MAX_STEPS."""
    state = _state(
        episode_step=jnp.int32(_MAX_STEPS - 1),
        reward=jnp.float32(0.0),
    )
    params = AtaraxParams(noop_max=0)
    new_state = _GAME._step(_KEY, state, jnp.int32(0), params)
    assert bool(new_state.done), (
        f"Expected done=True at episode_step={_MAX_STEPS}"
    )
