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

"""Kung-Fu Master — JAX-native game implementation.

Fight through five floors of a pagoda, punching and kicking enemies to
rescue Princess Silvia from the Devil King.

Action space (9 actions):
    0 — NOOP
    1 — FIRE  (punch)
    2 — UP    (jump)
    3 — RIGHT
    4 — DOWN  (crouch / kick low)
    5 — LEFT
    6 — RIGHT + FIRE
    7 — LEFT + FIRE
    8 — DOWN + FIRE (low kick)

Scoring:
    Enemy punched/kicked — +100
    Floor boss defeated  — +500
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_N_ENEMIES: int = 5
_PLAYER_Y: int = 170  # fixed y (single floor scroll)
_PLAYER_SPEED: float = 2.5
_JUMP_VEL: float = -5.5
_GRAVITY: float = 0.4
_PUNCH_RANGE: float = 22.0
_KICK_RANGE: float = 18.0
_ENEMY_SPEED: float = 1.0
_ATTACK_TIMER: int = 8
_INIT_LIVES: int = 3
_FLOOR_LENGTH: int = 5  # floors to clear

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([60, 40, 20], dtype=jnp.uint8)
_COLOR_FLOOR = jnp.array([100, 70, 30], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 180, 40], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 60, 60], dtype=jnp.uint8)
_COLOR_HIT = jnp.array([255, 255, 255], dtype=jnp.uint8)


@chex.dataclass
class KungFuMasterState(AtariState):
    """
    Complete Kung-Fu Master game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y (jumping or on floor).
    player_vy : jax.Array
        float32 — Vertical velocity.
    player_dir : jax.Array
        int32 — Facing direction (+1=right, -1=left).
    crouching : jax.Array
        bool — Player is crouching.
    attack_timer : jax.Array
        int32 — Frames of current attack animation.
    enemy_x : jax.Array
        float32[5] — Enemy x positions.
    enemy_hp : jax.Array
        int32[5] — Enemy hit points.
    enemy_active : jax.Array
        bool[5] — Enemy alive.
    spawn_timer : jax.Array
        int32 — Frames until next enemy spawn.
    floor : jax.Array
        int32 — Current floor (0=ground, 4=top).
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_vy: jax.Array
    player_dir: jax.Array
    crouching: jax.Array
    attack_timer: jax.Array
    enemy_x: jax.Array
    enemy_hp: jax.Array
    enemy_active: jax.Array
    spawn_timer: jax.Array
    floor: jax.Array
    key: jax.Array


class KungFuMaster(AtariEnv):
    """
    Kung-Fu Master implemented as a pure JAX function suite.

    Fight through all floors to rescue the princess.  Lives: 3.
    """

    num_actions: int = 9

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> KungFuMasterState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : KungFuMasterState
            Player at left of first floor, 3 lives.
        """
        return KungFuMasterState(
            player_x=jnp.float32(20.0),
            player_y=jnp.float32(float(_PLAYER_Y) - 14.0),
            player_vy=jnp.float32(0.0),
            player_dir=jnp.int32(1),
            crouching=jnp.bool_(False),
            attack_timer=jnp.int32(0),
            enemy_x=jnp.full(_N_ENEMIES, 160.0, dtype=jnp.float32),
            enemy_hp=jnp.ones(_N_ENEMIES, dtype=jnp.int32) * 3,
            enemy_active=jnp.zeros(_N_ENEMIES, dtype=jnp.bool_),
            spawn_timer=jnp.int32(60),
            floor=jnp.int32(0),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(
        self, state: KungFuMasterState, action: jax.Array
    ) -> KungFuMasterState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : KungFuMasterState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : KungFuMasterState
            State after one emulated frame.
        """
        key, sk = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Movement
        move_r = (action == jnp.int32(3)) | (action == jnp.int32(6))
        move_l = (action == jnp.int32(5)) | (action == jnp.int32(7))
        crouching = action == jnp.int32(4)
        new_dir = jnp.where(
            move_r, jnp.int32(1), jnp.where(move_l, jnp.int32(-1), state.player_dir)
        )

        new_px = jnp.clip(
            state.player_x
            + jnp.where(move_r & ~crouching, _PLAYER_SPEED, 0.0)
            + jnp.where(move_l & ~crouching, -_PLAYER_SPEED, 0.0),
            5.0,
            155.0,
        )

        # Jump
        floor_y = jnp.float32(_PLAYER_Y) - 14.0
        on_floor = state.player_y >= floor_y - 2.0
        do_jump = (action == jnp.int32(2)) & on_floor
        new_vy = jnp.where(do_jump, jnp.float32(_JUMP_VEL), state.player_vy + _GRAVITY)
        new_py = state.player_y + new_vy
        landed = new_py >= floor_y
        new_py = jnp.where(landed, floor_y, new_py)
        new_vy = jnp.where(landed, jnp.float32(0.0), new_vy)

        # Attack
        do_attack = (
            (action == jnp.int32(1))
            | (action == jnp.int32(6))
            | (action == jnp.int32(7))
            | (action == jnp.int32(8))
        )
        new_attack_timer = jnp.where(
            do_attack, jnp.int32(_ATTACK_TIMER), state.attack_timer - jnp.int32(1)
        )
        new_attack_timer = jnp.maximum(new_attack_timer, jnp.int32(0))
        attacking = new_attack_timer > jnp.int32(0)

        # Attack range (punch/kick)
        punch_x = new_px + new_dir.astype(jnp.float32) * _PUNCH_RANGE
        enemy_hit = (
            attacking
            & state.enemy_active
            & (jnp.abs(state.enemy_x - punch_x) < _PUNCH_RANGE)
        )
        n_hits = jnp.sum(enemy_hit).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_hits * 100)
        new_enemy_hp = state.enemy_hp - jnp.where(enemy_hit, jnp.int32(1), jnp.int32(0))
        new_enemy_active = state.enemy_active & (new_enemy_hp > jnp.int32(0))

        # Enemy movement (chase player)
        enemy_dx = jnp.sign(new_px - state.enemy_x) * _ENEMY_SPEED
        new_enemy_x = state.enemy_x + jnp.where(state.enemy_active, enemy_dx, 0.0)

        # Enemy touches player
        enemy_touches = new_enemy_active & (jnp.abs(new_enemy_x - new_px) < 12.0)
        hit = jnp.any(enemy_touches)
        new_lives = state.lives - jnp.where(hit, jnp.int32(1), jnp.int32(0))
        new_px = jnp.where(hit, jnp.float32(20.0), new_px)
        new_enemy_active = jnp.where(
            hit, jnp.zeros(_N_ENEMIES, dtype=jnp.bool_), new_enemy_active
        )
        new_enemy_x = jnp.where(
            hit, jnp.full(_N_ENEMIES, 160.0, dtype=jnp.float32), new_enemy_x
        )

        # Spawn enemy
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = new_spawn_timer <= jnp.int32(0)
        new_spawn_timer = jnp.where(do_spawn, jnp.int32(60), new_spawn_timer)
        free_slot = jnp.argmin(new_enemy_active.astype(jnp.int32))
        spawn_side = jax.random.randint(sk, (), 0, 2)
        spawn_x = jnp.where(spawn_side == 0, jnp.float32(155.0), jnp.float32(5.0))
        new_enemy_x = jnp.where(
            do_spawn, new_enemy_x.at[free_slot].set(spawn_x), new_enemy_x
        )
        new_enemy_active = jnp.where(
            do_spawn, new_enemy_active.at[free_slot].set(True), new_enemy_active
        )
        new_enemy_hp = jnp.where(
            do_spawn, new_enemy_hp.at[free_slot].set(jnp.int32(3)), new_enemy_hp
        )

        # Floor advance: player reaches right edge
        reached_end = new_px >= 155.0
        new_floor = state.floor + jnp.where(reached_end, jnp.int32(1), jnp.int32(0))
        step_reward = step_reward + jnp.where(
            reached_end, jnp.float32(500.0), jnp.float32(0.0)
        )
        new_px = jnp.where(reached_end, jnp.float32(5.0), new_px)

        done = (new_lives <= jnp.int32(0)) | (new_floor >= jnp.int32(_FLOOR_LENGTH))

        return KungFuMasterState(
            player_x=new_px,
            player_y=new_py,
            player_vy=new_vy,
            player_dir=new_dir,
            crouching=crouching,
            attack_timer=new_attack_timer,
            enemy_x=new_enemy_x,
            enemy_hp=new_enemy_hp,
            enemy_active=new_enemy_active,
            spawn_timer=new_spawn_timer,
            floor=new_floor,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: KungFuMasterState, action: jax.Array) -> KungFuMasterState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : KungFuMasterState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : KungFuMasterState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: KungFuMasterState) -> KungFuMasterState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: KungFuMasterState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : KungFuMasterState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Floor strip
        floor_mask = (_ROW_IDX >= _PLAYER_Y) & (_ROW_IDX <= _PLAYER_Y + 10)
        frame = jnp.where(floor_mask[:, :, None], _COLOR_FLOOR, frame)

        # Enemies
        def draw_enemy(frm, i):
            ex = state.enemy_x[i].astype(jnp.int32)
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= _PLAYER_Y - 22)
                & (_ROW_IDX <= _PLAYER_Y - 2)
                & (_COL_IDX >= ex - 8)
                & (_COL_IDX <= ex + 8)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        player_mask = (
            (_ROW_IDX >= py - 12)
            & (_ROW_IDX <= py + 12)
            & (_COL_IDX >= px - 6)
            & (_COL_IDX <= px + 6)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Kung-Fu Master action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_DOWN: 4,
            pygame.K_s: 4,
            pygame.K_LEFT: 5,
            pygame.K_a: 5,
        }
