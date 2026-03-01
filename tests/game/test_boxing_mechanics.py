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

"""Physics-level mechanics tests for Boxing.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_boxing_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.boxing import (
    Boxing,
    _CPU_START_X,
    _CPU_START_Y,
    _KO_SCORE,
    _PLAYER_SPEED,
    _PLAYER_START_X,
    _PLAYER_START_Y,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Boxing()
_INIT = _GAME._reset(_KEY)


def _state(**kw):
    """Return a fresh initial state with optional field overrides."""
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Action effects — player movement (representative subset of 18 actions)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "action,exp_dx,exp_dy",
    [
        pytest.param(0, 0.0, 0.0, id="noop"),
        pytest.param(1, 0.0, 0.0, id="fire"),              # FIRE alone: no movement
        pytest.param(2, 0.0, -_PLAYER_SPEED, id="up"),
        pytest.param(3, +_PLAYER_SPEED, 0.0, id="right"),
        pytest.param(4, -_PLAYER_SPEED, 0.0, id="left"),
        pytest.param(5, 0.0, +_PLAYER_SPEED, id="down"),
    ],
)
def test_action_moves_player(action, exp_dx, exp_dy):
    state = _state(reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(action))
    actual_dx = float(new_state.player_x) - float(state.player_x)
    actual_dy = float(new_state.player_y) - float(state.player_y)
    assert abs(actual_dx - exp_dx) < 1e-4, (
        f"action {action}: dx expected {exp_dx}, got {actual_dx}"
    )
    assert abs(actual_dy - exp_dy) < 1e-4, (
        f"action {action}: dy expected {exp_dy}, got {actual_dy}"
    )


# ---------------------------------------------------------------------------
# Punch mechanics
# ---------------------------------------------------------------------------
def test_punch_lands_when_in_range():
    """Player FIRE when CPU is within punch range → reward +1."""
    state = _state(
        player_x=jnp.float32(_PLAYER_START_X),
        player_y=jnp.float32(_PLAYER_START_Y),
        cpu_x=jnp.float32(_PLAYER_START_X),    # same x as player → manhattan=0
        cpu_y=jnp.float32(_PLAYER_START_Y),
        punch_timer=jnp.int32(0),
        cpu_punch_timer=jnp.int32(10),          # suppress CPU punch
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(1))  # FIRE
    assert abs(float(new_state.reward) - 1.0) < 1e-4


def test_punch_misses_when_out_of_range():
    """Player FIRE when CPU is far away (default start positions) → reward 0."""
    state = _state(
        player_x=jnp.float32(_PLAYER_START_X),
        player_y=jnp.float32(_PLAYER_START_Y),
        cpu_x=jnp.float32(_CPU_START_X),
        cpu_y=jnp.float32(_CPU_START_Y),
        punch_timer=jnp.int32(0),
        cpu_punch_timer=jnp.int32(10),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(1))  # FIRE
    assert abs(float(new_state.reward) - 0.0) < 1e-4


def test_cpu_punch_costs_negative_reward():
    """CPU auto-punches when in range with a ready timer → reward -1."""
    state = _state(
        player_x=jnp.float32(_PLAYER_START_X),
        player_y=jnp.float32(_PLAYER_START_Y),
        cpu_x=jnp.float32(_PLAYER_START_X),    # close to player
        cpu_y=jnp.float32(_PLAYER_START_Y),
        punch_timer=jnp.int32(10),              # player can't punch
        cpu_punch_timer=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert abs(float(new_state.reward) - (-1.0)) < 1e-4


def test_punch_cooldown_prevents_immediate_repunch():
    """When punch_timer > 0, player punch does not land."""
    state = _state(
        player_x=jnp.float32(_PLAYER_START_X),
        player_y=jnp.float32(_PLAYER_START_Y),
        cpu_x=jnp.float32(_PLAYER_START_X),
        cpu_y=jnp.float32(_PLAYER_START_Y),
        score=jnp.int32(5),
        punch_timer=jnp.int32(5),              # still in cooldown
        cpu_punch_timer=jnp.int32(10),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(1))  # FIRE
    assert abs(float(new_state.reward) - 0.0) < 1e-4, "Punch must not land during cooldown"
    assert int(new_state.score) == 5, "Score must be unchanged"


# ---------------------------------------------------------------------------
# Knockout (episode end)
# ---------------------------------------------------------------------------
def test_player_ko_ends_episode():
    """Player reaches 100 landed punches → knockout → done."""
    state = _state(
        player_x=jnp.float32(_PLAYER_START_X),
        player_y=jnp.float32(_PLAYER_START_Y),
        cpu_x=jnp.float32(_PLAYER_START_X),
        cpu_y=jnp.float32(_PLAYER_START_Y),
        score=jnp.int32(_KO_SCORE - 1),        # one punch away from KO
        punch_timer=jnp.int32(0),
        cpu_punch_timer=jnp.int32(10),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(1))  # FIRE
    assert bool(new_state.done)


def test_cpu_ko_ends_episode():
    """CPU reaches 100 landed punches → knockout → done."""
    state = _state(
        player_x=jnp.float32(_PLAYER_START_X),
        player_y=jnp.float32(_PLAYER_START_Y),
        cpu_x=jnp.float32(_PLAYER_START_X),
        cpu_y=jnp.float32(_PLAYER_START_Y),
        cpu_hits=jnp.int32(_KO_SCORE - 1),     # one CPU punch away from KO
        punch_timer=jnp.int32(10),
        cpu_punch_timer=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert bool(new_state.done)
