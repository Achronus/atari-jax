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

"""Physics-level mechanics tests for Space Invaders.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_space_invaders_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.space_invaders import (
    SpaceInvaders,
    _ALIEN_BULLET_SPEED,
    _ALIEN_COLS,
    _ALIEN_DROP_Y,
    _ALIEN_H,
    _ALIEN_INIT_X,
    _ALIEN_INIT_Y,
    _ALIEN_MOVE_INITIAL,
    _ALIEN_ROWS,
    _ALIEN_STEP_X,
    _ALIEN_W,
    _BULLET_H,
    _BULLET_W,
    _CANNON_H,
    _CANNON_SPEED,
    _CANNON_W,
    _CANNON_Y,
    _COL_STEP,
    _GROUND_Y,
    _PLAY_LEFT,
    _PLAY_RIGHT,
    _PLAYER_BULLET_SPEED,
    _ROW_SCORES,
    _ROW_STEP,
)

_KEY = jax.random.PRNGKey(0)
_GAME = SpaceInvaders()
_INIT = _GAME._reset(_KEY)

_PLAYER_INIT_X = float((_PLAY_LEFT + _PLAY_RIGHT - _CANNON_W) / 2)


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
        pytest.param(1, 0.0, id="fire_no_move"),
        pytest.param(2, _CANNON_SPEED, id="right"),
        pytest.param(3, -_CANNON_SPEED, id="left"),
        pytest.param(4, _CANNON_SPEED, id="rightfire"),
        pytest.param(5, -_CANNON_SPEED, id="leftfire"),
    ],
)
def test_action_moves_cannon(action, expected_delta):
    state = _state(reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(action))
    actual_delta = float(new_state.player_x) - float(state.player_x)
    assert abs(actual_delta - expected_delta) < 1e-4


# ---------------------------------------------------------------------------
# Alien formation movement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dx",
    [
        pytest.param(_ALIEN_STEP_X, id="right"),
        pytest.param(-_ALIEN_STEP_X, id="left"),
    ],
)
def test_formation_moves_on_timer(dx):
    # Use initial x=26; moving right: formation_right = 26+8+108=142 < 152 (no edge hit)
    # Moving left with alien_x=50: 50-8=42 > 8 (no edge hit)
    alien_x = 26.0 if dx > 0 else 50.0
    state = _state(
        alien_x=jnp.float32(alien_x),
        alien_dx=jnp.float32(dx),
        move_timer=jnp.int32(0),
        player_bullet_active=jnp.bool_(False),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    expected_x = alien_x + dx
    assert abs(float(new_state.alien_x) - expected_x) < 1e-3, (
        f"direction {dx:+.0f}: expected alien_x={expected_x}, got {float(new_state.alien_x)}"
    )


@pytest.mark.parametrize(
    "edge,alien_x,alien_dx",
    [
        # Right edge: alien_x + dx puts formation_right > PLAY_RIGHT
        pytest.param("right", 40.0, _ALIEN_STEP_X, id="right_edge"),
        # Left edge: alien_x + dx < PLAY_LEFT
        pytest.param("left", float(_PLAY_LEFT), -_ALIEN_STEP_X, id="left_edge"),
    ],
)
def test_formation_drops_and_reverses_on_edge(edge, alien_x, alien_dx):
    state = _state(
        alien_x=jnp.float32(alien_x),
        alien_dx=jnp.float32(alien_dx),
        move_timer=jnp.int32(0),
        player_bullet_active=jnp.bool_(False),
        reward=jnp.float32(0.0),
    )
    old_y = float(state.alien_y)
    old_dx = float(state.alien_dx)
    new_state = _GAME._step_physics(state, jnp.int32(0))
    # Formation drops
    assert abs(float(new_state.alien_y) - (old_y + _ALIEN_DROP_Y)) < 1e-3, (
        f"{edge} edge: alien_y should have dropped by {_ALIEN_DROP_Y}"
    )
    # Direction reverses
    assert float(new_state.alien_dx) * old_dx < 0, (
        f"{edge} edge: alien_dx should have flipped sign"
    )


# ---------------------------------------------------------------------------
# Formation acceleration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_alive,expected_interval",
    [
        pytest.param(55, max(1, 55 * _ALIEN_MOVE_INITIAL // (_ALIEN_ROWS * _ALIEN_COLS)), id="full_55"),
        pytest.param(27, max(1, 27 * _ALIEN_MOVE_INITIAL // (_ALIEN_ROWS * _ALIEN_COLS)), id="half_27"),
        pytest.param(11, max(1, 11 * _ALIEN_MOVE_INITIAL // (_ALIEN_ROWS * _ALIEN_COLS)), id="low_11"),
        pytest.param(1, 1, id="last_1"),
    ],
)
def test_move_interval_scales_with_alive_count(n_alive, expected_interval):
    # Build aliens grid with exactly n_alive active slots (row-major order)
    flat = jnp.arange(_ALIEN_ROWS * _ALIEN_COLS) < n_alive
    aliens = flat.reshape(_ALIEN_ROWS, _ALIEN_COLS)
    state = _state(
        aliens=aliens,
        alien_x=jnp.float32(_ALIEN_INIT_X),
        alien_dx=jnp.float32(_ALIEN_STEP_X),
        move_timer=jnp.int32(0),           # trigger move this frame
        player_bullet_active=jnp.bool_(False),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.move_timer) == expected_interval, (
        f"n_alive={n_alive}: expected interval {expected_interval}, "
        f"got {int(new_state.move_timer)}"
    )


# ---------------------------------------------------------------------------
# Player bullet kills alien — row score
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "row,expected_reward",
    [
        pytest.param(0, float(_ROW_SCORES[0]), id="row0_white"),
        pytest.param(1, float(_ROW_SCORES[1]), id="row1_yellow"),
        pytest.param(2, float(_ROW_SCORES[2]), id="row2_yellow"),
        pytest.param(3, float(_ROW_SCORES[3]), id="row3_green"),
        pytest.param(4, float(_ROW_SCORES[4]), id="row4_green"),
    ],
)
def test_kill_alien_row_score(row, expected_reward):
    # Alien (row, col=0) centre
    alien_left = _ALIEN_INIT_X                     # col 0 left edge
    alien_top = _ALIEN_INIT_Y + row * _ROW_STEP    # row r top edge
    # Position bullet so that after moving up by PLAYER_BULLET_SPEED it overlaps the alien
    pbx = alien_left                                           # x inside alien x-span
    pby = alien_top + _BULLET_H                               # y: after -SPEED still inside
    state = _state(
        player_bullet_x=jnp.float32(pbx),
        player_bullet_y=jnp.float32(pby),
        player_bullet_active=jnp.bool_(True),
        alien_bullet_active=jnp.bool_(False),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert abs(float(new_state.reward) - expected_reward) < 1e-3, (
        f"row {row}: expected reward {expected_reward}, got {float(new_state.reward)}"
    )


def test_killed_alien_is_deactivated():
    """Alien at (row=0, col=0) is removed from the grid after a bullet hit."""
    alien_left = _ALIEN_INIT_X
    alien_top = _ALIEN_INIT_Y
    pbx = alien_left
    pby = alien_top + _BULLET_H
    state = _state(
        player_bullet_x=jnp.float32(pbx),
        player_bullet_y=jnp.float32(pby),
        player_bullet_active=jnp.bool_(True),
        alien_bullet_active=jnp.bool_(False),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert not bool(new_state.aliens[0, 0]), "Alien (0, 0) should be deactivated after hit"


# ---------------------------------------------------------------------------
# Alien bullet hits cannon → life lost
# ---------------------------------------------------------------------------
def test_alien_bullet_hit_cannon_loses_life():
    # Position alien bullet directly over the cannon; after moving down it overlaps
    cannon_x = _PLAYER_INIT_X + 1.0          # inside cannon x-span
    ab_y = float(_CANNON_Y) - _ALIEN_BULLET_SPEED   # will land at CANNON_Y after 1 step
    state = _state(
        player_x=jnp.float32(_PLAYER_INIT_X),
        alien_bullet_x=jnp.float32(cannon_x),
        alien_bullet_y=jnp.float32(ab_y),
        alien_bullet_active=jnp.bool_(True),
        player_bullet_active=jnp.bool_(False),
        lives=jnp.int32(3),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.lives) == 2, (
        f"Expected lives=2 after cannon hit, got {int(new_state.lives)}"
    )


# ---------------------------------------------------------------------------
# Episode termination conditions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "condition",
    [
        pytest.param("all_dead", id="all_aliens_killed"),
        pytest.param("formation_ground", id="aliens_reach_ground"),
        pytest.param("lives_zero", id="lives_exhausted"),
    ],
)
def test_episode_termination(condition):
    if condition == "all_dead":
        state = _state(
            aliens=jnp.zeros((_ALIEN_ROWS, _ALIEN_COLS), dtype=jnp.bool_),
            player_bullet_active=jnp.bool_(False),
            alien_bullet_active=jnp.bool_(False),
            reward=jnp.float32(0.0),
        )
    elif condition == "formation_ground":
        # formation_bottom = alien_y + (ROWS-1)*ROW_STEP + ALIEN_H >= GROUND_Y
        alien_y = float(_GROUND_Y) - (_ALIEN_ROWS - 1) * _ROW_STEP - _ALIEN_H
        state = _state(
            alien_y=jnp.float32(alien_y),
            player_bullet_active=jnp.bool_(False),
            alien_bullet_active=jnp.bool_(False),
            reward=jnp.float32(0.0),
        )
    else:  # lives_zero
        cannon_x = _PLAYER_INIT_X + 1.0
        ab_y = float(_CANNON_Y) - _ALIEN_BULLET_SPEED
        state = _state(
            player_x=jnp.float32(_PLAYER_INIT_X),
            alien_bullet_x=jnp.float32(cannon_x),
            alien_bullet_y=jnp.float32(ab_y),
            alien_bullet_active=jnp.bool_(True),
            player_bullet_active=jnp.bool_(False),
            lives=jnp.int32(1),
            reward=jnp.float32(0.0),
        )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert bool(new_state.done), f"condition '{condition}': expected done=True"
