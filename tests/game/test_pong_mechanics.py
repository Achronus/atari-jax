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

"""Physics-level mechanics tests for Pong.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.  States are constructed
via ``game._reset(key).__replace__(**overrides)`` so the pytree shape is always
correct.

Run::

    pytest tests/game/test_pong_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.pong import (
    Pong,
    _BALL_H,
    _BALL_W,
    _CENTRE_X,
    _CENTRE_Y,
    _COURT_BOTTOM,
    _COURT_LEFT,
    _COURT_RIGHT,
    _COURT_TOP,
    _CPU_X,
    _PADDLE_H,
    _PADDLE_SPEED,
    _PADDLE_W,
    _PLAYER_X,
    _WIN_SCORE,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Pong()
_INIT = _GAME._reset(_KEY)


def _state(**kw):
    """Return a fresh initial state with optional field overrides."""
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Action effects — all 6 ALE actions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "action,expected_delta",
    [
        pytest.param(0, 0.0, id="noop"),
        pytest.param(1, 0.0, id="fire_noop"),
        pytest.param(2, -_PADDLE_SPEED, id="right_up"),
        pytest.param(3, _PADDLE_SPEED, id="left_down"),
        pytest.param(4, -_PADDLE_SPEED, id="rightfire_up"),
        pytest.param(5, _PADDLE_SPEED, id="leftfire_down"),
    ],
)
def test_action_moves_player_paddle(action, expected_delta):
    state = _state()
    new_state = _GAME._step_physics(state, jnp.int32(action))
    actual_delta = float(new_state.player_y) - float(state.player_y)
    assert abs(actual_delta - expected_delta) < 1e-4


# ---------------------------------------------------------------------------
# Wall bounces
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "side,ball_y,ball_dy",
    [
        pytest.param("top", float(_COURT_TOP) + 1.0, -2.0, id="top"),
        pytest.param("bottom", float(_COURT_BOTTOM - _BALL_H) - 1.0, 2.0, id="bottom"),
    ],
)
def test_wall_bounce_flips_dy(side, ball_y, ball_dy):
    state = _state(
        ball_x=jnp.float32(_CENTRE_X),
        ball_y=jnp.float32(ball_y),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(ball_dy),
        ball_active=jnp.bool_(True),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    # After a wall bounce the y-velocity must reverse sign
    assert float(new_state.ball_dy) * ball_dy < 0, (
        f"{side} wall: expected ball_dy to flip from {ball_dy}, "
        f"got {float(new_state.ball_dy)}"
    )


# ---------------------------------------------------------------------------
# Paddle deflections
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "side,ball_x,ball_dx,player_y,cpu_y",
    [
        # Player (right) paddle deflects rightward ball back left
        pytest.param(
            "player",
            float(_PLAYER_X) - 2.0,   # ball just left of paddle
            2.0,                        # moving rightward
            80.0,                       # paddle spans [80, 96)
            100.0,                      # cpu irrelevant here
            id="player",
        ),
        # CPU (left) paddle deflects leftward ball back right
        pytest.param(
            "cpu",
            float(_CPU_X + _PADDLE_W) + 0.0,   # ball just right of cpu paddle right edge
            -3.0,                                 # moving leftward
            100.0,                                # player irrelevant
            80.0,                                 # cpu spans [80, 96)
            id="cpu",
        ),
    ],
)
def test_paddle_deflect_reverses_dx(side, ball_x, ball_dx, player_y, cpu_y):
    state = _state(
        ball_x=jnp.float32(ball_x),
        ball_y=jnp.float32(85.0),   # inside both paddle y-ranges
        ball_dx=jnp.float32(ball_dx),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        player_y=jnp.float32(player_y),
        cpu_y=jnp.float32(cpu_y),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    # After deflection the x-velocity must reverse
    assert float(new_state.ball_dx) * ball_dx < 0, (
        f"{side} paddle: expected ball_dx to flip from {ball_dx}, "
        f"got {float(new_state.ball_dx)}"
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ball_x,ball_dx,score_field",
    [
        # Ball exits left → player scores
        pytest.param(5.0, -3.0, "score", id="player_scores"),
        # Ball exits right → cpu scores
        pytest.param(float(_COURT_RIGHT - _BALL_W) + 1.0, 3.0, "opp_score", id="cpu_scores"),
    ],
)
def test_ball_exit_increments_score(ball_x, ball_dx, score_field):
    state = _state(
        ball_x=jnp.float32(ball_x),
        ball_y=jnp.float32(160.0),   # far below any paddle, no deflection
        ball_dx=jnp.float32(ball_dx),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        player_y=jnp.float32(34.0),   # paddle at top of court, away from ball
        cpu_y=jnp.float32(34.0),
        score=jnp.int32(0),
        opp_score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(getattr(new_state, score_field)) == 1


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------
def test_episode_ends_when_score_reaches_21():
    # Inject score=20, ball exits left → score becomes 21 → done
    state = _state(
        ball_x=jnp.float32(2.0),
        ball_y=jnp.float32(160.0),
        ball_dx=jnp.float32(-3.0),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        player_y=jnp.float32(34.0),
        cpu_y=jnp.float32(34.0),
        score=jnp.int32(_WIN_SCORE - 1),
        opp_score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.score) == _WIN_SCORE
    assert bool(new_state.done)


def test_no_termination_at_score_20():
    # score reaches 20 (<21) — episode continues
    state = _state(
        ball_x=jnp.float32(2.0),
        ball_y=jnp.float32(160.0),
        ball_dx=jnp.float32(-3.0),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        player_y=jnp.float32(34.0),
        cpu_y=jnp.float32(34.0),
        score=jnp.int32(_WIN_SCORE - 2),
        opp_score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.score) == _WIN_SCORE - 1
    assert not bool(new_state.done)
