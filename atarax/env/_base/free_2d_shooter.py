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

"""Template 2 — Free 2D Shooter base classes.

Games in this template (Asteroids, Gravitar, …):

- Ship has full rotational freedom and inertial velocity
- Thrust accelerates in the facing direction; drag decelerates
- Screen wraps toroidally (x and y both modulo world size)
- Projectiles are fired from the ship tip into a fixed-size pool

Concrete games inherit `Free2DShooterState` to add game-specific fields,
and `Free2DShooterGame` to gain shared inertial physics helpers.
"""

import chex
import jax.numpy as jnp

from atarax.game import AtaraxGame
from atarax.state import AtariState


@chex.dataclass
class Free2DShooterState(AtariState):
    """
    Shared state for Template 2 Free 2D Shooter games.

    Inherits `reward`, `done`, `step`, `episode_step`, `lives`,
    `score`, `level`, and `key` from `~atarax.state.AtariState`.

    Concrete game states (e.g. `AsteroidsState`) inherit from this class
    and add any game-specific fields on top.

    Parameters
    ----------
    ship_x : chex.Array
        float32 scalar — ship centre x in world coordinates.
    ship_y : chex.Array
        float32 scalar — ship centre y in world coordinates.
    ship_vx : chex.Array
        float32 scalar — ship horizontal velocity (pixels per step).
    ship_vy : chex.Array
        float32 scalar — ship vertical velocity; positive = moving downward.
    ship_angle : chex.Array
        float32 scalar — heading in radians; `0` = pointing right.
        Increases clockwise (screen y-axis points down).
    ship_alive : chex.Array
        bool scalar — `False` during the death/respawn sequence.
    fire_cooldown : chex.Array
        int32 scalar — frames remaining until the next shot is allowed.
    thrust_active : chex.Array
        bool scalar — `True` on the frame thrust is applied (for flame render).
    invincible : chex.Array
        int32 scalar — frames of post-respawn invincibility remaining.
    """

    ship_x: chex.Array
    ship_y: chex.Array
    ship_vx: chex.Array
    ship_vy: chex.Array
    ship_angle: chex.Array
    ship_alive: chex.Array
    fire_cooldown: chex.Array
    thrust_active: chex.Array
    invincible: chex.Array


class Free2DShooterGame(AtaraxGame):
    """
    Abstract base class for Template 2 Free 2D Shooter games.

    Provides shared, branch-free physics helpers that all T2 games can reuse.
    Concrete games inherit this class and implement `_reset`,
    `_step`, and `render`.
    """

    def _apply_thrust(
        self,
        vx: chex.Array,
        vy: chex.Array,
        angle: chex.Array,
        thrust: chex.Array,
        drag: chex.Array,
        max_speed: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        """
        Apply thrust in the heading direction then dampen with drag.

        Thrust adds `(cos(angle), sin(angle)) * thrust` to velocity.
        Drag subtracts `drag * velocity` (linear friction). The resulting
        speed is clamped to `max_speed`.

        Parameters
        ----------
        vx : chex.Array
            float32 scalar — current horizontal velocity.
        vy : chex.Array
            float32 scalar — current vertical velocity.
        angle : chex.Array
            float32 scalar — ship heading in radians.
        thrust : chex.Array
            float32 scalar — velocity magnitude added per step.
        drag : chex.Array
            float32 scalar — fractional velocity reduction per step.
        max_speed : chex.Array
            float32 scalar — speed is clamped to this value after thrust.

        Returns
        -------
        new_vx : chex.Array
            float32 scalar — updated horizontal velocity.
        new_vy : chex.Array
            float32 scalar — updated vertical velocity.
        """
        new_vx = vx + jnp.cos(angle) * thrust - drag * vx
        new_vy = vy + jnp.sin(angle) * thrust - drag * vy
        speed = jnp.sqrt(new_vx**2 + new_vy**2)
        scale = jnp.minimum(
            jnp.float32(1.0), max_speed / jnp.maximum(speed, jnp.float32(1e-6))
        )
        return new_vx * scale, new_vy * scale

    def _wrap_torus(
        self,
        x: chex.Array,
        y: chex.Array,
        world_w: chex.Array,
        world_h: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        """
        Apply toroidal (wraparound) boundary conditions.

        Entities that move past the right edge reappear at the left, and
        vice versa; same for top/bottom.

        Parameters
        ----------
        x : chex.Array
            float32 scalar or array — x position(s) to wrap.
        y : chex.Array
            float32 scalar or array — y position(s) to wrap.
        world_w : chex.Array
            float32 scalar — world width; wraps in `[0, world_w)`.
        world_h : chex.Array
            float32 scalar — world height; wraps in `[0, world_h)`.

        Returns
        -------
        wrapped_x : chex.Array
            float32 — x position(s) mapped to `[0, world_w)`.
        wrapped_y : chex.Array
            float32 — y position(s) mapped to `[0, world_h)`.
        """
        return x % world_w, y % world_h

    def _move_bullets(
        self,
        bullets: chex.Array,
        world_w: chex.Array,
        world_h: chex.Array,
    ) -> chex.Array:
        """
        Advance all bullets in the pool by their velocity and decrement lifetime.

        Bullets wrap toroidally. Any bullet whose lifetime reaches zero is
        deactivated via arithmetic gate (no branching).

        Parameters
        ----------
        bullets : chex.Array
            (N, 6) float32 — bullet pool `[x, y, vx, vy, lifetime, active]`.
        world_w : chex.Array
            float32 scalar — world width for toroidal wrap.
        world_h : chex.Array
            float32 scalar — world height for toroidal wrap.

        Returns
        -------
        updated : chex.Array
            (N, 6) float32 — pool after movement and lifetime decrement.
        """
        active = bullets[:, 5]
        new_x = (bullets[:, 0] + bullets[:, 2] * active) % world_w
        new_y = (bullets[:, 1] + bullets[:, 3] * active) % world_h
        new_life = bullets[:, 4] - active
        new_active = active * (new_life > jnp.float32(0.0)).astype(jnp.float32)
        return jnp.stack(
            [new_x, new_y, bullets[:, 2], bullets[:, 3], new_life, new_active], axis=1
        )

    def _fire_bullet(
        self,
        bullets: chex.Array,
        ship_x: chex.Array,
        ship_y: chex.Array,
        angle: chex.Array,
        bullet_speed: chex.Array,
        tip_offset: chex.Array,
        lifetime: chex.Array,
        fire_cooldown: chex.Array,
        max_cooldown: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        """
        Spawn a bullet from the ship tip if the cooldown has expired.

        Finds the first inactive slot in the pool using `argmin` on the
        active column — branch-free and JIT-safe. If no slot is free, the
        fire action is silently ignored.

        Parameters
        ----------
        bullets : chex.Array
            (N, 6) float32 — bullet pool `[x, y, vx, vy, lifetime, active]`.
        ship_x : chex.Array
            float32 scalar — ship centre x.
        ship_y : chex.Array
            float32 scalar — ship centre y.
        angle : chex.Array
            float32 scalar — ship heading in radians.
        bullet_speed : chex.Array
            float32 scalar — bullet speed in pixels per step.
        tip_offset : chex.Array
            float32 scalar — distance from ship centre to spawn point.
        lifetime : chex.Array
            int32 scalar — steps before bullet is deactivated.
        fire_cooldown : chex.Array
            int32 scalar — current cooldown; bullet fires only when `<= 0`.
        max_cooldown : chex.Array
            int32 scalar — cooldown reset value after firing.

        Returns
        -------
        new_bullets : chex.Array
            (N, 6) float32 — updated bullet pool with new bullet inserted.
        new_cooldown : chex.Array
            int32 scalar — updated fire cooldown.
        """
        can_fire = fire_cooldown <= 0
        slot = jnp.argmin(bullets[:, 5])
        slot_free = bullets[slot, 5] < jnp.float32(0.5)

        spawn_x = ship_x + jnp.cos(angle) * tip_offset
        spawn_y = ship_y + jnp.sin(angle) * tip_offset
        bvx = jnp.cos(angle) * bullet_speed
        bvy = jnp.sin(angle) * bullet_speed
        new_bullet = jnp.array(
            [spawn_x, spawn_y, bvx, bvy, lifetime.astype(jnp.float32), jnp.float32(1.0)]
        )
        do_fire = can_fire & slot_free
        new_bullets = jnp.where(do_fire, bullets.at[slot].set(new_bullet), bullets)
        new_cooldown = jnp.where(do_fire, max_cooldown, fire_cooldown - 1)
        return new_bullets, new_cooldown
