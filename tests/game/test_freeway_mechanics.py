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

"""Physics-level mechanics tests for Freeway.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.  Episode-termination
tests call ``game._step()`` because ``done`` is set there (after incrementing
``episode_step``).

Run::

    pytest tests/game/test_freeway_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.game import AtaraxParams
from atarax.games.freeway import (
    Freeway,
    _CAR_LANE_Y,
    _CHICKEN_START_Y,
    _CHICKEN_X,
    _GOAL_Y,
    _MAX_STEPS,
    _N_LANES,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Freeway()
_INIT = _GAME._reset(_KEY)
_PARAMS = AtaraxParams()


def _state(**kw):
    """Return a fresh initial state with optional field overrides."""
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Action effects — chicken movement (3 actions)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "action,expected_dy",
    [
        pytest.param(0, 0.0, id="noop"),
        pytest.param(1, -1.0, id="up"),
        pytest.param(2, +1.0, id="down"),
    ],
)
def test_action_moves_chicken(action, expected_dy):
    # Cars off-screen (x=200) so no collision; chicken in middle of road.
    car_x = jnp.full((_N_LANES,), 200.0, dtype=jnp.float32)
    state = _state(
        chicken_y=jnp.float32(100.0),
        car_x=car_x,
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(action))
    actual_dy = float(new_state.chicken_y) - float(state.chicken_y)
    assert abs(actual_dy - expected_dy) < 1e-4


# ---------------------------------------------------------------------------
# Goal scoring
# ---------------------------------------------------------------------------
def test_reaching_goal_increments_score():
    """Chicken one step from goal: UP action reaches goal → score +1."""
    car_x = jnp.full((_N_LANES,), 200.0, dtype=jnp.float32)
    state = _state(
        chicken_y=jnp.float32(float(_GOAL_Y) + 1.0),
        car_x=car_x,
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(1))  # UP
    assert int(new_state.score) == 1


def test_reaching_goal_resets_chicken_y():
    """After reaching the goal the chicken returns to the start row."""
    car_x = jnp.full((_N_LANES,), 200.0, dtype=jnp.float32)
    state = _state(
        chicken_y=jnp.float32(float(_GOAL_Y) + 1.0),
        car_x=car_x,
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(1))  # UP
    assert abs(float(new_state.chicken_y) - float(_CHICKEN_START_Y)) < 1e-4


# ---------------------------------------------------------------------------
# Car collision
# ---------------------------------------------------------------------------
def test_car_collision_pushes_chicken_back():
    """Car overlaps the chicken → chicken is pushed back by one lane."""
    # Lane 9 (bottom): _CAR_LANE_Y[9] = 182.  Place car at chicken x.
    lane_y = float(_CAR_LANE_Y[9])
    car_x = jnp.array([200.0] * 9 + [float(_CHICKEN_X)], dtype=jnp.float32)
    state = _state(
        chicken_y=jnp.float32(lane_y),
        car_x=car_x,
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert float(new_state.chicken_y) > float(state.chicken_y), (
        f"Chicken should be pushed back: before={float(state.chicken_y)}, "
        f"after={float(new_state.chicken_y)}"
    )


# ---------------------------------------------------------------------------
# Episode termination (done set in _step, not _step_physics)
# ---------------------------------------------------------------------------
def test_episode_ends_at_max_steps():
    """Last agent step triggers done."""
    car_x = jnp.full((_N_LANES,), 200.0, dtype=jnp.float32)
    state = _state(
        car_x=car_x,
        episode_step=jnp.int32(_MAX_STEPS - 1),
        done=jnp.bool_(False),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step(_KEY, state, jnp.int32(0), _PARAMS)
    assert bool(new_state.done)


def test_no_termination_before_max_steps():
    """Step before the last does not set done."""
    car_x = jnp.full((_N_LANES,), 200.0, dtype=jnp.float32)
    state = _state(
        car_x=car_x,
        episode_step=jnp.int32(_MAX_STEPS - 2),
        done=jnp.bool_(False),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step(_KEY, state, jnp.int32(0), _PARAMS)
    assert not bool(new_state.done)
