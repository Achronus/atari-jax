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

"""Physics-level mechanics tests for Breakout.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_breakout_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.breakout import (
    Breakout,
    _BALL_SIZE,
    _BRICK_COLS,
    _BRICK_H,
    _BRICK_ROWS,
    _BRICK_W,
    _BRICK_X0,
    _BRICK_Y0,
    _PADDLE_H,
    _PADDLE_SPEED,
    _PADDLE_W,
    _PADDLE_Y,
    _PLAY_BOTTOM,
    _PLAY_LEFT,
    _PLAY_RIGHT,
    _ROW_SCORES,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Breakout()
_INIT = _GAME._reset(_KEY)

_PADDLE_INIT_X = float((_PLAY_LEFT + _PLAY_RIGHT) / 2 - _PADDLE_W / 2)


def _state(**kw):
    """Return a fresh initial state with optional field overrides."""
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Action effects — paddle movement (4 actions)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "action,expected_delta",
    [
        pytest.param(0, 0.0, id="noop"),
        pytest.param(1, 0.0, id="fire_no_paddle"),
        pytest.param(2, _PADDLE_SPEED, id="right"),
        pytest.param(3, -_PADDLE_SPEED, id="left"),
    ],
)
def test_action_moves_paddle(action, expected_delta):
    state = _state(reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(action))
    actual_delta = float(new_state.paddle_x) - float(state.paddle_x)
    assert abs(actual_delta - expected_delta) < 1e-4


# ---------------------------------------------------------------------------
# Ball activation
# ---------------------------------------------------------------------------
def test_fire_activates_ball():
    state = _state(ball_active=jnp.bool_(False), reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(1))  # FIRE
    assert bool(new_state.ball_active)


def test_inactive_ball_stays_on_paddle_without_fire():
    state = _state(ball_active=jnp.bool_(False), reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert not bool(new_state.ball_active)


# ---------------------------------------------------------------------------
# Brick hit scoring — one test per row
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "row,expected_score",
    [
        pytest.param(0, float(_ROW_SCORES[0]), id="row0_red"),
        pytest.param(1, float(_ROW_SCORES[1]), id="row1_red"),
        pytest.param(2, float(_ROW_SCORES[2]), id="row2_orange"),
        pytest.param(3, float(_ROW_SCORES[3]), id="row3_orange"),
        pytest.param(4, float(_ROW_SCORES[4]), id="row4_green"),
        pytest.param(5, float(_ROW_SCORES[5]), id="row5_green"),
    ],
)
def test_brick_hit_score_by_row(row, expected_score):
    # Position ball inside the target brick (col 0) with zero velocity so it stays put.
    ball_x = float(_BRICK_X0) + 1.0
    ball_y = float(_BRICK_Y0) + row * float(_BRICK_H) + 1.0
    state = _state(
        ball_x=jnp.float32(ball_x),
        ball_y=jnp.float32(ball_y),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert abs(float(new_state.reward) - expected_score) < 1e-4, (
        f"row {row}: expected reward {expected_score}, got {float(new_state.reward)}"
    )


# ---------------------------------------------------------------------------
# Brick removed after hit — one per row
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "row",
    [pytest.param(r, id=f"row{r}") for r in range(_BRICK_ROWS)],
)
def test_brick_removed_after_hit(row):
    ball_x = float(_BRICK_X0) + 1.0
    ball_y = float(_BRICK_Y0) + row * float(_BRICK_H) + 1.0
    state = _state(
        ball_x=jnp.float32(ball_x),
        ball_y=jnp.float32(ball_y),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    # Brick at (row, 0) must be cleared
    assert not bool(new_state.bricks[row, 0]), (
        f"row {row}: brick was not removed after ball hit"
    )


# ---------------------------------------------------------------------------
# Speed tier bumps on upper-row hit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "row",
    [pytest.param(0, id="row0"), pytest.param(1, id="row1")],
)
def test_upper_row_hit_bumps_speed_tier(row):
    ball_x = float(_BRICK_X0) + 1.0
    ball_y = float(_BRICK_Y0) + row * float(_BRICK_H) + 1.0
    state = _state(
        ball_x=jnp.float32(ball_x),
        ball_y=jnp.float32(ball_y),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        speed_tier=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.speed_tier) >= 1, (
        f"row {row}: speed_tier did not increase from 0 (got {int(new_state.speed_tier)})"
    )


# ---------------------------------------------------------------------------
# Life loss and episode termination
# ---------------------------------------------------------------------------
def test_ball_exit_bottom_loses_life():
    state = _state(
        ball_x=jnp.float32(80.0),
        ball_y=jnp.float32(float(_PLAY_BOTTOM) + 1.0),   # already below boundary
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(2.0),
        ball_active=jnp.bool_(True),
        lives=jnp.int32(3),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.lives) == 2
    assert not bool(new_state.ball_active)


def test_lives_zero_ends_episode():
    state = _state(
        ball_x=jnp.float32(80.0),
        ball_y=jnp.float32(float(_PLAY_BOTTOM) + 1.0),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(2.0),
        ball_active=jnp.bool_(True),
        lives=jnp.int32(1),   # last life
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.lives) == 0
    assert bool(new_state.done)


def test_lives_nonzero_continues():
    state = _state(
        ball_x=jnp.float32(80.0),
        ball_y=jnp.float32(float(_PLAY_BOTTOM) + 1.0),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(2.0),
        ball_active=jnp.bool_(True),
        lives=jnp.int32(3),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.lives) == 2
    assert not bool(new_state.done)
