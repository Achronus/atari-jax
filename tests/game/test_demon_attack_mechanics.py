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

"""Physics-level mechanics tests for Demon Attack.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_demon_attack_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.demon_attack import (
    DemonAttack,
    _BULLET_H,
    _BULLET_SPEED,
    _BULLET_W,
    _CANNON_H,
    _CANNON_W,
    _CANNON_Y,
    _DEMON_H,
    _DEMON_INIT_X,
    _DEMON_INIT_Y,
    _DEMON_W,
    _PLAYER_SPEED,
)

_KEY = jax.random.PRNGKey(0)
_GAME = DemonAttack()
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
        pytest.param(4, +_PLAYER_SPEED, id="rightfire"),
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
# Bullet mechanics
# ---------------------------------------------------------------------------
def test_fire_spawns_bullet():
    """FIRE when no bullet active → bullet_active=True."""
    state = _state(bullet_active=jnp.bool_(False), reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(1))  # FIRE
    assert bool(new_state.bullet_active)


def test_bullet_kills_wave0_demon_scores_37():
    """Wave 0: bullet hitting demon → score += 37 (_KILL_BASE_SCORE)."""
    ex = float(_DEMON_INIT_X[0])
    ey = float(_DEMON_INIT_Y[0])
    # After move: new_by = initial_by - BULLET_SPEED; need new_by + BULLET_H > ey (strictly)
    initial_by = ey - float(_BULLET_H) + float(_BULLET_SPEED) + 1.0
    state = _state(
        bullet_x=jnp.float32(ex + 2.0),
        bullet_y=jnp.float32(initial_by),
        bullet_active=jnp.bool_(True),
        wave=jnp.int32(0),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.score) == 37, (
        f"Expected score 37 on wave 0, got {int(new_state.score)}"
    )


def test_bullet_kills_wave1_demon_scores_74():
    """Wave 1: bullet hitting demon → score += 74 (2 * _KILL_BASE_SCORE)."""
    ex = float(_DEMON_INIT_X[0])
    ey = float(_DEMON_INIT_Y[0])
    # After move: new_by = initial_by - BULLET_SPEED; need new_by + BULLET_H > ey (strictly)
    initial_by = ey - float(_BULLET_H) + float(_BULLET_SPEED) + 1.0
    state = _state(
        bullet_x=jnp.float32(ex + 2.0),
        bullet_y=jnp.float32(initial_by),
        bullet_active=jnp.bool_(True),
        wave=jnp.int32(1),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.score) == 74, (
        f"Expected score 74 on wave 1, got {int(new_state.score)}"
    )


# ---------------------------------------------------------------------------
# Enemy bullet vs player
# ---------------------------------------------------------------------------
def test_enemy_bullet_hits_player_loses_life():
    """Enemy bullet hitting player cannon → lives -= 1."""
    bx = 78.0  # inside player cannon [76, 76+13]
    initial_by = float(_CANNON_Y) - float(_BULLET_H) + 1.0
    state = _state(
        player_x=jnp.float32(76.0),
        enemy_bullet_x=jnp.array([bx, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_y=jnp.array([initial_by, 0.0, 0.0], dtype=jnp.float32),
        enemy_bullet_active=jnp.array([True, False, False], dtype=jnp.bool_),
        lives=jnp.int32(3),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.lives) == 2, (
        f"Expected lives=2, got {int(new_state.lives)}"
    )


# ---------------------------------------------------------------------------
# Episode end
# ---------------------------------------------------------------------------
def test_episode_ends_when_lives_zero():
    """When lives reach 0 → done."""
    enemy_y = _DEMON_INIT_Y.at[0].set(jnp.float32(float(_CANNON_Y) - 0.05))
    state = _state(
        demon_y=enemy_y,
        lives=jnp.int32(1),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert bool(new_state.done)
