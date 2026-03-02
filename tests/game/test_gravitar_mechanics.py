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

"""Gravitar mechanics tests — direct _step_physics calls (one emulated frame).

Each test exercises a single physics rule in isolation.  All assertions use the
_INIT state as a baseline and call _step_physics once so results are exact.
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.gravitar import (
    Gravitar,
    _ASTRONAUT_RESCUE_DY,
    _ASTRONAUT_REWARD,
    _ASTRONAUT_XS,
    _ASTRONAUT_Y,
    _BUNKER_REWARD,
    _BUNKER_XS,
    _BUNKER_Y,
    _GROUND_Y,
    _INIT_LIVES,
    _N_ASTRONAUTS,
    _N_BULLETS,
    _N_BUNKERS,
    _ROTATE_RATE,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Gravitar()
_INIT = _GAME._reset(_KEY)


def _state(**kw):
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Ship movement
# ---------------------------------------------------------------------------


def test_thrust_accelerates_ship_upward():
    """Thrust at angle=0 (pointing up): net force is upward (thrust > gravity)."""
    state = _state()  # angle=0, ship_vy=0
    new = _GAME._step_physics(state, jnp.int32(2))  # UP = thrust
    assert float(new.ship_vy) < 0.0


def test_noop_ship_falls_due_to_gravity():
    """NOOP: gravity pulls the ship downward (vy increases)."""
    state = _state()
    new = _GAME._step_physics(state, jnp.int32(0))
    assert float(new.ship_vy) > 0.0


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotate_cw_increases_angle():
    """RIGHT (action 3) rotates the ship clockwise by _ROTATE_RATE."""
    state = _state()
    new = _GAME._step_physics(state, jnp.int32(3))
    assert float(new.ship_angle) == pytest.approx(_ROTATE_RATE)


def test_rotate_ccw_decreases_angle():
    """LEFT (action 4) rotates the ship counter-clockwise by _ROTATE_RATE."""
    state = _state()
    new = _GAME._step_physics(state, jnp.int32(4))
    assert float(new.ship_angle) == pytest.approx(-_ROTATE_RATE)


# ---------------------------------------------------------------------------
# Bullets
# ---------------------------------------------------------------------------


def test_fire_spawns_bullet():
    """FIRE with no active bullets spawns one bullet in slot 0."""
    state = _state()  # all bullets inactive
    new = _GAME._step_physics(state, jnp.int32(1))  # FIRE
    assert bool(new.bullet_active[0])


def test_fire_does_not_spawn_when_slots_full():
    """FIRE when all bullet slots are active does not spawn an extra bullet."""
    # Bullets moving upward (vy=-1) from y=50 stay in bounds after one frame (49>0).
    state = _state(
        bullet_x=jnp.full(_N_BULLETS, 80.0, dtype=jnp.float32),
        bullet_y=jnp.full(_N_BULLETS, 50.0, dtype=jnp.float32),
        bullet_active=jnp.ones(_N_BULLETS, dtype=jnp.bool_),
        bullet_timer=jnp.full(_N_BULLETS, 50, dtype=jnp.int32),
        bullet_vx=jnp.zeros(_N_BULLETS, dtype=jnp.float32),
        bullet_vy=jnp.full(_N_BULLETS, -1.0, dtype=jnp.float32),
    )
    new = _GAME._step_physics(state, jnp.int32(1))
    # All 3 slots remain active (none expire in one frame with timer=50)
    assert int(jnp.sum(new.bullet_active.astype(jnp.int32))) == _N_BULLETS


# ---------------------------------------------------------------------------
# Bullet vs bunker collision
# ---------------------------------------------------------------------------


def test_bullet_kills_bunker():
    """A downward-moving bullet reaching bunker 0 destroys it (+250 score).

    Bullet starts at y=BUNKER_Y with bvy=1.0 (moving toward ground).
    After one frame new_by=176.0, within the ±8 DY window; bvy>0 satisfied.
    """
    state = _state(
        bullet_x=jnp.array([_BUNKER_XS[0], -10.0, -10.0], dtype=jnp.float32),
        bullet_y=jnp.array([_BUNKER_Y, -10.0, -10.0], dtype=jnp.float32),
        bullet_vx=jnp.zeros(_N_BULLETS, dtype=jnp.float32),
        bullet_vy=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        bullet_active=jnp.array([True, False, False], dtype=jnp.bool_),
        bullet_timer=jnp.array([50, 0, 0], dtype=jnp.int32),
    )
    new = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert not bool(new.bunker_active[0])
    assert int(new.score) == _BUNKER_REWARD


def test_bullet_deactivated_after_kill():
    """The bullet that kills a bunker is removed from active slots."""
    state = _state(
        bullet_x=jnp.array([_BUNKER_XS[0], -10.0, -10.0], dtype=jnp.float32),
        bullet_y=jnp.array([_BUNKER_Y, -10.0, -10.0], dtype=jnp.float32),
        bullet_vx=jnp.zeros(_N_BULLETS, dtype=jnp.float32),
        bullet_vy=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        bullet_active=jnp.array([True, False, False], dtype=jnp.bool_),
        bullet_timer=jnp.array([50, 0, 0], dtype=jnp.int32),
    )
    new = _GAME._step_physics(state, jnp.int32(0))
    assert not bool(new.bullet_active[0])


# ---------------------------------------------------------------------------
# Astronaut rescue
# ---------------------------------------------------------------------------


def test_astronaut_rescue():
    """Ship within rescue range of astronaut 0 rescues it (+1000 score)."""
    ax = float(_ASTRONAUT_XS[0])
    # Place ship slightly above the rescue threshold
    state = _state(
        ship_x=jnp.float32(ax),
        ship_y=jnp.float32(_ASTRONAUT_Y - _ASTRONAUT_RESCUE_DY + 1.0),
        ship_vx=jnp.float32(0.0),
        ship_vy=jnp.float32(0.0),
    )
    new = _GAME._step_physics(state, jnp.int32(0))
    assert bool(new.astronaut_rescued[0])
    assert int(new.score) == _ASTRONAUT_REWARD


def test_already_rescued_astronaut_not_double_scored():
    """Passing over an already-rescued astronaut gives no extra reward."""
    ax = float(_ASTRONAUT_XS[0])
    all_rescued = jnp.ones(_N_ASTRONAUTS, dtype=jnp.bool_)
    state = _state(
        ship_x=jnp.float32(ax),
        ship_y=jnp.float32(_ASTRONAUT_Y - _ASTRONAUT_RESCUE_DY + 1.0),
        ship_vx=jnp.float32(0.0),
        ship_vy=jnp.float32(0.0),
        astronaut_rescued=all_rescued,
    )
    new = _GAME._step_physics(state, jnp.int32(0))
    assert int(new.score) == 0


# ---------------------------------------------------------------------------
# Ground collision (life loss and shield)
# ---------------------------------------------------------------------------


def test_crash_without_shield_loses_life():
    """Ship hitting the ground without shield loses a life."""
    state = _state(
        ship_y=jnp.float32(_GROUND_Y - 1.0),
        ship_vy=jnp.float32(2.0),
    )
    new = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new.lives) == _INIT_LIVES - 1


def test_shield_prevents_ground_crash():
    """Shield (action 5 = DOWN) prevents life loss on ground impact."""
    state = _state(
        ship_y=jnp.float32(_GROUND_Y - 1.0),
        ship_vy=jnp.float32(2.0),
    )
    new = _GAME._step_physics(state, jnp.int32(5))  # DOWN = shield
    assert int(new.lives) == _INIT_LIVES


def test_shield_bounces_ship_off_ground():
    """After a shielded ground hit, ship velocity becomes upward."""
    state = _state(
        ship_y=jnp.float32(_GROUND_Y - 1.0),
        ship_vy=jnp.float32(2.0),
    )
    new = _GAME._step_physics(state, jnp.int32(5))
    assert float(new.ship_vy) < 0.0


# ---------------------------------------------------------------------------
# Fuel
# ---------------------------------------------------------------------------


def test_fuel_empty_loses_life():
    """fuel=0: even without a crash, fuel_out triggers a life loss."""
    state = _state(fuel=jnp.int32(0))
    new = _GAME._step_physics(state, jnp.int32(0))  # NOOP
    assert int(new.lives) == _INIT_LIVES - 1


def test_thrust_drains_fuel():
    """One frame of thrust consumes one unit of fuel."""
    from atarax.games.gravitar import _FUEL_DRAIN, _FUEL_MAX

    state = _state()  # fuel=_FUEL_MAX
    new = _GAME._step_physics(state, jnp.int32(2))  # UP = thrust
    assert int(new.fuel) == _FUEL_MAX - _FUEL_DRAIN


# ---------------------------------------------------------------------------
# Wave completion
# ---------------------------------------------------------------------------


def test_wave_complete_respawns_bunkers():
    """Destroying the last active bunker resets all bunkers and increments wave."""
    state = _state(
        bunker_active=jnp.array(
            [True, False, False, False, False], dtype=jnp.bool_
        ),
        bullet_x=jnp.array([_BUNKER_XS[0], -10.0, -10.0], dtype=jnp.float32),
        bullet_y=jnp.array([_BUNKER_Y, -10.0, -10.0], dtype=jnp.float32),
        bullet_vx=jnp.zeros(_N_BULLETS, dtype=jnp.float32),
        bullet_vy=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        bullet_active=jnp.array([True, False, False], dtype=jnp.bool_),
        bullet_timer=jnp.array([50, 0, 0], dtype=jnp.int32),
    )
    new = _GAME._step_physics(state, jnp.int32(0))
    assert jnp.all(new.bunker_active)
    assert int(new.level) == 1
    assert int(new.score) == _BUNKER_REWARD


# ---------------------------------------------------------------------------
# Terminal condition
# ---------------------------------------------------------------------------


def test_terminal_when_lives_zero():
    """done=True when lives reach zero."""
    state = _state(lives=jnp.int32(0))
    new = _GAME._step_physics(state, jnp.int32(0))
    assert bool(new.done)


def test_not_terminal_with_lives():
    """done=False while lives > 0 and no fatal event occurs."""
    state = _state(lives=jnp.int32(3))
    new = _GAME._step_physics(state, jnp.int32(0))
    assert not bool(new.done)
