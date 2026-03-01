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

"""Physics-level mechanics tests for Tennis.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_tennis_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.tennis import (
    Tennis,
    _COURT_BOTTOM,
    _COURT_TOP,
    _PLAYER_SPEED,
    _WIN_GAMES,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Tennis()
_INIT = _GAME._reset(_KEY)


def _state(**kw):
    """Return a fresh initial state with optional field overrides."""
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Action effects — player movement (representative subset of 18 actions)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "action,exp_dy,exp_dx",
    [
        pytest.param(0, 0.0, 0.0, id="noop"),
        pytest.param(2, -_PLAYER_SPEED, 0.0, id="up"),
        pytest.param(3, 0.0, +_PLAYER_SPEED, id="right"),
        pytest.param(4, 0.0, -_PLAYER_SPEED, id="left"),
        pytest.param(5, +_PLAYER_SPEED, 0.0, id="down"),
        pytest.param(6, -_PLAYER_SPEED, +_PLAYER_SPEED, id="upright"),
        pytest.param(10, -_PLAYER_SPEED, 0.0, id="upfire"),
    ],
)
def test_action_moves_player(action, exp_dy, exp_dx):
    state = _state(reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(action))
    actual_dy = float(new_state.player_y) - float(state.player_y)
    actual_dx = float(new_state.player_x) - float(state.player_x)
    assert abs(actual_dy - exp_dy) < 1e-4, (
        f"action {action}: dy expected {exp_dy}, got {actual_dy}"
    )
    assert abs(actual_dx - exp_dx) < 1e-4, (
        f"action {action}: dx expected {exp_dx}, got {actual_dx}"
    )


# ---------------------------------------------------------------------------
# Scoring: ball exits court
# ---------------------------------------------------------------------------
def _ball_exits_top_state(**extra):
    """State with ball one pixel inside the CPU baseline, moving up."""
    return _state(
        ball_x=jnp.float32(75.0),
        ball_y=jnp.float32(float(_COURT_TOP) + 1.0),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(-2.0),
        ball_active=jnp.bool_(True),
        reward=jnp.float32(0.0),
        **extra,
    )


def _ball_exits_bottom_state(**extra):
    """State with ball one pixel inside the player baseline, moving down."""
    return _state(
        ball_x=jnp.float32(75.0),
        ball_y=jnp.float32(float(_COURT_BOTTOM) - 1.0),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(2.0),
        ball_active=jnp.bool_(True),
        reward=jnp.float32(0.0),
        **extra,
    )


def test_ball_exit_top_awards_player_point():
    """Ball exits past CPU baseline → player scores → reward +1."""
    new_state = _GAME._step_physics(_ball_exits_top_state(), jnp.int32(0))
    assert abs(float(new_state.reward) - 1.0) < 1e-4


def test_ball_exit_bottom_awards_cpu_point():
    """Ball exits past player baseline → CPU scores → reward -1."""
    new_state = _GAME._step_physics(_ball_exits_bottom_state(), jnp.int32(0))
    assert abs(float(new_state.reward) - (-1.0)) < 1e-4


# ---------------------------------------------------------------------------
# Point progression (love → 15 → 30 → 40)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "from_pts,to_pts",
    [
        pytest.param(0, 1, id="love_to_15"),
        pytest.param(1, 2, id="15_to_30"),
        pytest.param(2, 3, id="30_to_40"),
    ],
)
def test_point_scoring_increments_pts(from_pts, to_pts):
    state = _ball_exits_top_state(
        player_pts=jnp.int32(from_pts),
        cpu_pts=jnp.int32(0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.player_pts) == to_pts


# ---------------------------------------------------------------------------
# Deuce / advantage
# ---------------------------------------------------------------------------
def test_point_at_40_40_does_not_end_game():
    """At deuce (40–40), scoring gives advantage but does not win the game."""
    state = _ball_exits_top_state(
        player_pts=jnp.int32(3),  # 40
        cpu_pts=jnp.int32(3),     # 40
        score=jnp.int32(0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.player_pts) == 4, "Expected advantage (pts=4)"
    assert int(new_state.score) == 0, "Score (games) must not change at deuce"


def test_advantage_then_win_ends_game():
    """At advantage (pts=4 vs 3), scoring again wins the game."""
    state = _ball_exits_top_state(
        player_pts=jnp.int32(4),  # advantage
        cpu_pts=jnp.int32(3),     # 40
        score=jnp.int32(0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.score) == 1, "Game should be won (score +1)"


# ---------------------------------------------------------------------------
# Episode end: first to _WIN_GAMES games wins
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("winner", ["player", "cpu"])
def test_six_games_ends_episode(winner):
    if winner == "player":
        state = _ball_exits_top_state(
            score=jnp.int32(_WIN_GAMES - 1),  # 5 games won
            player_pts=jnp.int32(4),           # advantage → will win game
            cpu_pts=jnp.int32(3),
            cpu_games=jnp.int32(0),
        )
    else:
        state = _ball_exits_bottom_state(
            cpu_games=jnp.int32(_WIN_GAMES - 1),  # 5 games won
            cpu_pts=jnp.int32(4),                  # advantage → will win game
            player_pts=jnp.int32(3),
            score=jnp.int32(0),
        )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert bool(new_state.done), (
        f"Episode should end when {winner} reaches {_WIN_GAMES} games"
    )
