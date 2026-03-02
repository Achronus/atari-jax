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

"""Physics-level mechanics tests for Gopher.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_gopher_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.gopher import (
    Gopher,
    _BULLET_H,
    _BULLET_SPEED,
    _BULLET_W,
    _CARROT_H,
    _CARROT_X,
    _CARROT_Y,
    _GOPHER_ABOVE_Y,
    _GOPHER_H,
    _GOPHER_POINTS,
    _GOPHER_W,
    _N_CARROTS,
    _PLAYER_SPEED,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Gopher()
_INIT = _GAME._reset(_KEY)


def _state(**kw):
    """Return a fresh initial state with optional field overrides."""
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Action effects — player movement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "action,exp_dx",
    [
        pytest.param(0, 0.0, id="noop"),
        pytest.param(3, +_PLAYER_SPEED, id="right"),
        pytest.param(4, -_PLAYER_SPEED, id="left"),
        pytest.param(6, +_PLAYER_SPEED, id="rightfire"),
    ],
)
def test_action_moves_player(action, exp_dx):
    state = _state(player_x=jnp.float32(76.0), reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(action))
    actual_dx = float(new_state.player_x) - 76.0
    assert abs(actual_dx - exp_dx) < 1e-4, (
        f"action {action}: dx expected {exp_dx}, got {actual_dx}"
    )


# ---------------------------------------------------------------------------
# Shoot gopher
# ---------------------------------------------------------------------------
def test_bullet_kills_gopher_scores_200():
    """Bullet overlapping above-ground gopher → reward += 200."""
    # Gopher at (30, 50) above ground, not digging
    gx = 30.0
    gy = float(_GOPHER_ABOVE_Y)
    # Place bullet so it overlaps gopher after moving up
    initial_by = gy + float(_BULLET_SPEED) - float(_BULLET_H) + 1.0
    state = _state(
        bullet_x=jnp.float32(gx + 2.0),
        bullet_y=jnp.float32(initial_by),
        bullet_active=jnp.bool_(True),
        gopher_x=jnp.float32(gx),
        gopher_y=jnp.float32(gy),
        gopher_alive=jnp.bool_(True),
        gopher_digging=jnp.bool_(False),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.score) == _GOPHER_POINTS, (
        f"Expected score {_GOPHER_POINTS} after killing gopher, got {int(new_state.score)}"
    )
    assert not bool(new_state.bullet_active), "Bullet should be deactivated after hit"


# ---------------------------------------------------------------------------
# Carrot theft
# ---------------------------------------------------------------------------
def test_gopher_steals_carrot_decrements_lives():
    """Gopher digging down to carrot y → carrot stolen, lives -= 1."""
    # Gopher targeting carrot 1 (middle, x=80), already at carrot x, just above carrot y
    carrot_idx = 1
    gx = float(_CARROT_X[carrot_idx])
    gy = float(_CARROT_Y) - 1.0  # will cross _CARROT_Y after descent
    state = _state(
        gopher_x=jnp.float32(gx),
        gopher_y=jnp.float32(gy),
        gopher_alive=jnp.bool_(True),
        gopher_target=jnp.int32(carrot_idx),
        gopher_digging=jnp.bool_(True),  # already digging
        carrot_alive=jnp.ones(_N_CARROTS, dtype=jnp.bool_),
        lives=jnp.int32(_N_CARROTS),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert not bool(new_state.carrot_alive[carrot_idx]), (
        f"Carrot {carrot_idx} should be stolen"
    )
    assert int(new_state.lives) == _N_CARROTS - 1, (
        f"Expected lives={_N_CARROTS - 1}, got {int(new_state.lives)}"
    )


# ---------------------------------------------------------------------------
# Episode end
# ---------------------------------------------------------------------------
def test_episode_ends_when_all_carrots_stolen():
    """All carrots stolen → done."""
    # One carrot left, gopher about to steal it
    carrot_alive = jnp.zeros(_N_CARROTS, dtype=jnp.bool_).at[0].set(jnp.bool_(True))
    gx = float(_CARROT_X[0])
    gy = float(_CARROT_Y) - 1.0
    state = _state(
        gopher_x=jnp.float32(gx),
        gopher_y=jnp.float32(gy),
        gopher_alive=jnp.bool_(True),
        gopher_target=jnp.int32(0),
        gopher_digging=jnp.bool_(True),
        carrot_alive=carrot_alive,
        lives=jnp.int32(1),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert bool(new_state.done), "Episode should end when all carrots are stolen"
