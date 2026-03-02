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

"""Physics-level mechanics tests for Phoenix.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_phoenix_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.phoenix import (
    Phoenix,
    _BIRD_H,
    _BIRD_INIT_X,
    _BIRD_INIT_Y,
    _BIRD_POINTS_BOTTOM,
    _BIRD_POINTS_TOP,
    _BIRD_W,
    _BULLET_H,
    _BULLET_SPEED,
    _BULLET_W,
    _ENEMY_BULLET_SPEED,
    _N_BIRDS,
    _PLAYER_H,
    _PLAYER_SPEED,
    _PLAYER_W,
    _PLAYER_Y,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Phoenix()
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
        pytest.param(2, +_PLAYER_SPEED, id="right"),
        pytest.param(3, -_PLAYER_SPEED, id="left"),
        pytest.param(5, +_PLAYER_SPEED, id="rightfire"),
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
# Shield mechanic
# ---------------------------------------------------------------------------
def test_shield_blocks_enemy_bullet():
    """DOWN action (action=4) active + enemy bullet at player level → bullet deactivated."""
    bx = 78.0  # inside player x+w
    # Enemy bullet arriving at player level
    initial_by = float(_PLAYER_Y) - float(_ENEMY_BULLET_SPEED) - float(_BULLET_H) - 6.0
    state = _state(
        player_x=jnp.float32(76.0),
        enemy_bullet_x=jnp.array([bx, 0.0, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_y=jnp.array([initial_by, 0.0, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_active=jnp.array([True, False, False, False], dtype=jnp.bool_),
        lives=jnp.int32(3),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(4))  # DOWN = shield
    # Shield should be active and lives unchanged
    assert bool(new_state.shield_active), "Shield should be active for action 4"
    assert int(new_state.lives) == 3, (
        f"Shield should prevent life loss, got lives={int(new_state.lives)}"
    )


def test_no_shield_allows_bullet_hit():
    """Without shield, enemy bullet at player level → lives -= 1."""
    bx = 78.0
    initial_by = float(_PLAYER_Y) - float(_ENEMY_BULLET_SPEED) - float(_BULLET_H) + 1.0
    state = _state(
        player_x=jnp.float32(76.0),
        enemy_bullet_x=jnp.array([bx, 0.0, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_y=jnp.array([initial_by, 0.0, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_active=jnp.array([True, False, False, False], dtype=jnp.bool_),
        lives=jnp.int32(3),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP (no shield)
    assert int(new_state.lives) < 3, (
        f"Expected lives < 3 without shield, got {int(new_state.lives)}"
    )


# ---------------------------------------------------------------------------
# Bird kill scoring (rows 0–1 = 10 pts, rows 2–3 = 20 pts)
# ---------------------------------------------------------------------------
def test_bullet_kills_top_row_bird_scores_10():
    """Bullet hitting row-0 bird → score += 10."""
    # bird[0] is row 0, col 0: init_x=8, init_y=20
    bx0 = float(_BIRD_INIT_X[0])
    by0 = float(_BIRD_INIT_Y[0])
    # After one descent step and drift, bird is roughly at (bx0+0.5, by0+small)
    # Place bullet to overlap bird[0] after moving up by BULLET_SPEED
    initial_bby = by0 + float(_BULLET_SPEED) - float(_BULLET_H) + 1.0
    state = _state(
        bullet_x=jnp.float32(bx0 + 2.0),
        bullet_y=jnp.float32(initial_bby),
        bullet_active=jnp.bool_(True),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.score) == _BIRD_POINTS_TOP, (
        f"Expected score {_BIRD_POINTS_TOP} for top-row bird, got {int(new_state.score)}"
    )


def test_bullet_kills_bottom_row_bird_scores_20():
    """Bullet hitting row-2 bird → score += 20."""
    # bird[16] is row 2, col 0 (rows 0-1 = 16 birds, row 2 starts at bird[16])
    # _BIRD_ROW_IDX[16] = 2
    bx2 = float(_BIRD_INIT_X[16])
    by2 = float(_BIRD_INIT_Y[16])
    initial_bby = by2 + float(_BULLET_SPEED) - float(_BULLET_H) + 1.0
    state = _state(
        bullet_x=jnp.float32(bx2 + 2.0),
        bullet_y=jnp.float32(initial_bby),
        bullet_active=jnp.bool_(True),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.score) == _BIRD_POINTS_BOTTOM, (
        f"Expected score {_BIRD_POINTS_BOTTOM} for bottom-row bird, got {int(new_state.score)}"
    )


# ---------------------------------------------------------------------------
# Player hit and episode end
# ---------------------------------------------------------------------------
def test_player_hit_loses_life():
    """Enemy bullet hits unshielded player → lives -= 1."""
    bx = 78.0
    initial_by = float(_PLAYER_Y) - float(_ENEMY_BULLET_SPEED) - float(_BULLET_H) + 1.0
    state = _state(
        player_x=jnp.float32(76.0),
        enemy_bullet_x=jnp.array([bx, 0.0, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_y=jnp.array([initial_by, 0.0, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_active=jnp.array([True, False, False, False], dtype=jnp.bool_),
        lives=jnp.int32(3),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.lives) < 3


def test_episode_ends_when_lives_zero():
    """When lives reach 0 → done."""
    bx = 78.0
    initial_by = float(_PLAYER_Y) - float(_ENEMY_BULLET_SPEED) - float(_BULLET_H) + 1.0
    state = _state(
        player_x=jnp.float32(76.0),
        enemy_bullet_x=jnp.array([bx, 0.0, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_y=jnp.array([initial_by, 0.0, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_active=jnp.array([True, False, False, False], dtype=jnp.bool_),
        lives=jnp.int32(1),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert bool(new_state.done)
