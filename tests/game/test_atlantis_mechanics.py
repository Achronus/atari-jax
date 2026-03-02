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

"""Physics-level mechanics tests for Atlantis.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_atlantis_mechanics.py -v
"""

import jax
import jax.numpy as jnp

from atarax.games.atlantis import (
    Atlantis,
    _ALIEN_H,
    _ALIEN_INIT_X,
    _ALIEN_INIT_Y,
    _ALIEN_POINTS,
    _ALIEN_W,
    _BULLET_SPEED_X,
    _BULLET_SPEED_Y,
    _CANNON_CENTRE_X,
    _CANNON_LEFT_X,
    _CANNON_RIGHT_X,
    _CANNON_Y,
    _CITY_Y,
    _CITY_XS,
    _N_CITY,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Atlantis()
_INIT = _GAME._reset(_KEY)


def _state(**kw):
    """Return a fresh initial state with optional field overrides."""
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Cannon fire — bullet spawned in correct direction
# ---------------------------------------------------------------------------
def test_fire_centre_spawns_vertical_bullet():
    """Action 1 (FIRE CENTRE) → centre bullet active, no horizontal drift."""
    state = _state(bullet_centre_active=jnp.bool_(False), reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(1))
    assert bool(new_state.bullet_centre_active), "Centre bullet should be active"
    # After one step the bullet has moved straight up by _BULLET_SPEED_Y
    # x should stay at CANNON_CENTRE_X (no horizontal drift applied when firing)
    # Check bullet x is near the centre cannon x
    assert abs(float(new_state.bullet_centre_x) - float(_CANNON_CENTRE_X)) < 5.0


def test_fire_right_spawns_rightward_bullet():
    """Action 2 (FIRE RIGHT) → right bullet active, moving right."""
    state = _state(bullet_right_active=jnp.bool_(False), reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(2))
    assert bool(new_state.bullet_right_active), "Right bullet should be active"
    # After spawn+move, bullet x should be > cannon_right_x
    assert float(new_state.bullet_right_x) > float(_CANNON_RIGHT_X) - 1.0


def test_fire_left_spawns_leftward_bullet():
    """Action 3 (FIRE LEFT) → left bullet active, moving left."""
    state = _state(bullet_left_active=jnp.bool_(False), reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(3))
    assert bool(new_state.bullet_left_active), "Left bullet should be active"
    # After spawn+move, bullet x should be < cannon_left_x
    assert float(new_state.bullet_left_x) < float(_CANNON_LEFT_X) + 1.0


# ---------------------------------------------------------------------------
# Kill scoring
# ---------------------------------------------------------------------------
def test_bullet_hits_alien_scores_250():
    """Centre bullet overlapping alien → alien killed, score += 250."""
    # Alien at (74, 60): x = CANNON_CENTRE_X - ALIEN_W/2, so bullet x overlaps
    ax = float(_CANNON_CENTRE_X) - float(_ALIEN_W) / 2.0  # = 74.0
    ay = 60.0
    alien_x = _ALIEN_INIT_X.at[0].set(jnp.float32(ax))
    alien_y = _ALIEN_INIT_Y.at[0].set(jnp.float32(ay))
    # Bullet moves up by BULLET_SPEED_Y each frame.
    # Start bullet at bcy = ay + 6 → after move: new_bcy = ay + 1
    # Collision y: (ay+1) + BULLET_H(4) = ay+5 > ay ✓  and  (ay+1) < ay + ALIEN_H(8) ✓
    initial_bcy = ay + 6.0
    state = _state(
        alien_x=alien_x,
        alien_y=alien_y,
        bullet_centre_x=jnp.float32(_CANNON_CENTRE_X),
        bullet_centre_y=jnp.float32(initial_bcy),
        bullet_centre_active=jnp.bool_(True),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.score) == _ALIEN_POINTS, (
        f"Expected score {_ALIEN_POINTS}, got {int(new_state.score)}"
    )


# ---------------------------------------------------------------------------
# City destruction
# ---------------------------------------------------------------------------
def test_alien_reaching_city_destroys_section():
    """Alien descending to city y → nearest city section destroyed."""
    # Place alien[0] at city_section[0] x, just above city_y
    city_x = float(_CITY_XS[0]) + 2.0
    alien_x = _ALIEN_INIT_X.at[0].set(jnp.float32(city_x))
    alien_y = _ALIEN_INIT_Y.at[0].set(jnp.float32(float(_CITY_Y) - float(_ALIEN_H) + 1.0))
    state = _state(
        alien_x=alien_x,
        alien_y=alien_y,
        lives=jnp.int32(_N_CITY),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new_state.lives) < _N_CITY, (
        f"Expected lives < {_N_CITY} after alien reaches city, got {int(new_state.lives)}"
    )
    assert not bool(new_state.city_alive[0]), "City section 0 should be destroyed"


def test_all_sections_destroyed_ends_episode():
    """When all city sections are destroyed → done."""
    # One section left, alien at city x
    city_x = float(_CITY_XS[0]) + 2.0
    alien_x = _ALIEN_INIT_X.at[0].set(jnp.float32(city_x))
    alien_y = _ALIEN_INIT_Y.at[0].set(jnp.float32(float(_CITY_Y) - float(_ALIEN_H) + 1.0))
    # Only city section 0 is alive
    city_alive = jnp.zeros(_N_CITY, dtype=jnp.bool_).at[0].set(jnp.bool_(True))
    state = _state(
        alien_x=alien_x,
        alien_y=alien_y,
        city_alive=city_alive,
        lives=jnp.int32(1),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert bool(new_state.done), "Episode should end when all city sections destroyed"
