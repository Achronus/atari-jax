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

"""Physics-level mechanics tests for Assault.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_assault_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.assault import (
    Assault,
    _BULLET_H,
    _BULLET_SPEED,
    _BULLET_W,
    _CANNON_H,
    _CANNON_W,
    _CANNON_Y,
    _ENEMY_BULLET_SPEED,
    _ENEMY_H,
    _ENEMY_INIT_X,
    _ENEMY_INIT_Y,
    _ENEMY_POINTS,
    _ENEMY_W,
    _N_ENEMIES,
    _PLAYER_LEFT,
    _PLAYER_RIGHT,
    _PLAYER_SPEED,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Assault()
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
        pytest.param(5, +_PLAYER_SPEED, id="rightfire"),
    ],
)
def test_action_moves_player(action, exp_dx):
    state = _state(player_x=jnp.float32(80.0), reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(action))
    actual_dx = float(new_state.player_x) - 80.0
    assert abs(actual_dx - exp_dx) < 1e-4, (
        f"action {action}: dx expected {exp_dx}, got {actual_dx}"
    )


# ---------------------------------------------------------------------------
# Bullet mechanics
# ---------------------------------------------------------------------------
def test_fire_spawns_bullet():
    """FIRE when no bullet active → bullet_active=True."""
    state = _state(bullet_active=jnp.bool_(False), reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(1))  # FIRE
    assert bool(new_state.bullet_active)


def test_bullet_kills_enemy_scores_10():
    """Bullet overlapping enemy[0] → enemy dies, score += 10."""
    # Row-0 enemies are at y=30 == _PLAY_TOP, so the bullet would go OOB after
    # moving up and be deactivated before collision.  Place enemy[0] at y=40
    # (between row-0 y=30 and row-1 y=50, unique position) so bullet stays valid.
    # Also row-1 enemies at y=50 share x=20 with enemy[0], so y=50 would give score=20.
    ex = float(_ENEMY_INIT_X[0])
    ey = 40.0  # between rows — no other enemy at this y
    enemy_y = _ENEMY_INIT_Y.at[0].set(jnp.float32(ey))
    # After move: new_by = initial_by - BULLET_SPEED = ey - BULLET_H + 1 = 37
    # OOB: 37 >= PLAY_TOP=30 ✓; Collision: 37+4=41 > 40 ✓
    initial_by = ey - float(_BULLET_H) + float(_BULLET_SPEED) + 1.0
    state = _state(
        enemy_y=enemy_y,
        bullet_x=jnp.float32(ex + 2.0),
        bullet_y=jnp.float32(initial_by),
        bullet_active=jnp.bool_(True),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.score) == _ENEMY_POINTS, (
        f"Expected score {_ENEMY_POINTS}, got {int(new_state.score)}"
    )
    assert not bool(new_state.bullet_active), "Bullet should be deactivated after hit"
    assert not bool(new_state.enemy_alive[0]), "Enemy 0 should be dead"


# ---------------------------------------------------------------------------
# Enemy bullet vs player
# ---------------------------------------------------------------------------
def test_player_hit_by_enemy_bullet_loses_life():
    """Enemy bullet hitting player → lives -= 1."""
    # Place enemy bullet so it hits the player cannon after one step
    # Player at x=80, cannon box: [80, 80+13] × [177, 185]
    # Enemy bullet moves down by _ENEMY_BULLET_SPEED per frame
    # Place bullet at player_x + 2, just above cannon_y
    bx = 82.0
    initial_by = float(_CANNON_Y) - float(_ENEMY_BULLET_SPEED) - float(_BULLET_H) + 1.0
    state = _state(
        player_x=jnp.float32(80.0),
        enemy_bullet_x=jnp.array([bx, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_y=jnp.array([initial_by, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_active=jnp.array([True, False, False], dtype=jnp.bool_),
        lives=jnp.int32(4),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.lives) == 3, (
        f"Expected lives=3 after hit, got {int(new_state.lives)}"
    )


# ---------------------------------------------------------------------------
# Enemy reaching bottom
# ---------------------------------------------------------------------------
def test_enemy_reaching_bottom_loses_life():
    """Enemy reaching cannon y → enemy removed, lives -= 1."""
    # Place enemy[0] one step above cannon_y so it crosses the threshold
    enemy_y = _ENEMY_INIT_Y.at[0].set(jnp.float32(_CANNON_Y - 0.05))
    state = _state(
        enemy_y=enemy_y,
        lives=jnp.int32(4),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.lives) < 4, (
        f"Expected lives < 4 after enemy reaches bottom, got {int(new_state.lives)}"
    )
    assert not bool(new_state.enemy_alive[0]), "Enemy 0 should be removed after reaching cannon"


# ---------------------------------------------------------------------------
# Episode end
# ---------------------------------------------------------------------------
def test_episode_ends_when_lives_zero():
    """When lives reach 0 → done."""
    # Set lives=1, enemy at cannon level → lives hits 0
    enemy_y = _ENEMY_INIT_Y.at[0].set(jnp.float32(_CANNON_Y - 0.05))
    state = _state(
        enemy_y=enemy_y,
        lives=jnp.int32(1),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert bool(new_state.done)
