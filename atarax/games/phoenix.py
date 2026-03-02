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

"""Phoenix — JAX-native game implementation.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Player   : y = 175, x ∈ [5, 147]
    Birds    : 4 rows × 8 cols; starts x=8, y=20; col spacing=16, row spacing=12

Action space (8 actions — ALE minimal set):
    0  NOOP
    1  FIRE
    2  RIGHT
    3  LEFT
    4  DOWN (shield)
    5  RIGHTFIRE
    6  LEFTFIRE
    7  DOWNFIRE
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# Geometry
_PLAYER_Y: int = 175
_PLAYER_LEFT: float = 5.0
_PLAYER_RIGHT: float = 147.0
_PLAYER_W: int = 13
_PLAYER_H: int = 8
_PLAYER_SPEED: float = 2.0

# Bullets
_BULLET_W: int = 1
_BULLET_H: int = 4
_BULLET_SPEED: float = 4.0
_ENEMY_BULLET_SPEED: float = 1.5

# Bird formation
_N_ROWS: int = 4
_N_COLS: int = 8
_N_BIRDS: int = _N_ROWS * _N_COLS  # 32
_BIRD_W: int = 10
_BIRD_H: int = 8
_COL_SPACING: int = 16
_ROW_SPACING: int = 12
_BIRD_X0: float = 8.0
_BIRD_Y0: float = 20.0

_BIRD_POINTS_TOP = 10  # rows 0–1
_BIRD_POINTS_BOTTOM = 20  # rows 2–3

# Bird drift speed (horizontal oscillation)
_BIRD_DRIFT_X: float = 0.5
_BIRD_DESCENT: float = 0.03  # slow descent per frame

_N_ENEMY_BULLETS: int = 4
_FIRE_INTERVAL: int = 30
_INIT_LIVES: int = 3
_FRAME_SKIP: int = 4
_PLAY_TOP: int = 15

# Precomputed initial bird positions
_BIRD_ROW_IDX = jnp.repeat(
    jnp.arange(_N_ROWS), _N_COLS
)  # [0,0,...,1,1,...,2,2,...,3,3,...]
_BIRD_COL_IDX = jnp.tile(jnp.arange(_N_COLS), _N_ROWS)  # [0,1,...,7, 0,1,...,7, ...]
_BIRD_INIT_X = _BIRD_X0 + _BIRD_COL_IDX.astype(jnp.float32) * _COL_SPACING
_BIRD_INIT_Y = _BIRD_Y0 + _BIRD_ROW_IDX.astype(jnp.float32) * _ROW_SPACING

# Alternating oscillation directions
_BIRD_INIT_DX = jnp.where(
    _BIRD_COL_IDX % 2 == 0,
    jnp.float32(_BIRD_DRIFT_X),
    jnp.float32(-_BIRD_DRIFT_X),
)

# Per-bird point value: rows 0-1 = 10, rows 2-3 = 20
_BIRD_POINTS = jnp.where(
    _BIRD_ROW_IDX < 2, jnp.int32(_BIRD_POINTS_TOP), jnp.int32(_BIRD_POINTS_BOTTOM)
)

# Render
_ROW_IDX_R = jnp.arange(210)[:, None]
_COL_IDX_R = jnp.arange(160)[None, :]

_PLAYER_COLOR = jnp.array([100, 200, 100], dtype=jnp.uint8)
_SHIELD_COLOR = jnp.array([80, 80, 220], dtype=jnp.uint8)
_BULLET_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)
_BIRD_TOP_COLOR = jnp.array([220, 180, 60], dtype=jnp.uint8)
_BIRD_BOT_COLOR = jnp.array([200, 60, 200], dtype=jnp.uint8)
_ENEMY_BULLET_COLOR = jnp.array([255, 80, 80], dtype=jnp.uint8)


@chex.dataclass
class PhoenixState(AtariState):
    """
    Complete Phoenix game state.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player left-edge x ∈ [5, 147].
    bullet_x : jax.Array
        float32 — Player bullet x.
    bullet_y : jax.Array
        float32 — Player bullet y.
    bullet_active : jax.Array
        bool — True while player bullet in flight.
    shield_active : jax.Array
        bool — True while DOWN is held.
    bird_x : jax.Array
        float32[32] — Bird x positions.
    bird_y : jax.Array
        float32[32] — Bird y positions.
    bird_dx : jax.Array
        float32[32] — Bird horizontal oscillation velocities.
    bird_alive : jax.Array
        bool[32] — Active birds.
    enemy_bullet_x : jax.Array
        float32[4] — Enemy bullet x slots.
    enemy_bullet_y : jax.Array
        float32[4] — Enemy bullet y slots.
    enemy_bullet_active : jax.Array
        bool[4] — Active enemy bullets.
    wave : jax.Array
        int32 — Current wave; formation speed increases each wave.
    fire_timer : jax.Array
        int32 — Countdown to next enemy shot.
    """

    player_x: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_active: chex.Array
    shield_active: chex.Array
    bird_x: chex.Array
    bird_y: chex.Array
    bird_dx: chex.Array
    bird_alive: chex.Array
    enemy_bullet_x: chex.Array
    enemy_bullet_y: chex.Array
    enemy_bullet_active: chex.Array
    wave: chex.Array
    fire_timer: chex.Array


class Phoenix(AtaraxGame):
    """
    Phoenix implemented as a pure-JAX function suite.

    Shoot waves of birds that dive from the top of the screen.  A shield
    can block enemy fire briefly.  Clear all birds to advance to the next wave.
    """

    num_actions: int = 8

    def _reset(self, key: chex.PRNGKey) -> PhoenixState:
        """Return the canonical initial game state."""
        return PhoenixState(
            player_x=jnp.float32(76.0),
            bullet_x=jnp.float32(0.0),
            bullet_y=jnp.float32(0.0),
            bullet_active=jnp.bool_(False),
            shield_active=jnp.bool_(False),
            bird_x=_BIRD_INIT_X.copy(),
            bird_y=_BIRD_INIT_Y.copy(),
            bird_dx=_BIRD_INIT_DX.copy(),
            bird_alive=jnp.ones(_N_BIRDS, dtype=jnp.bool_),
            enemy_bullet_x=jnp.zeros(_N_ENEMY_BULLETS, dtype=jnp.float32),
            enemy_bullet_y=jnp.zeros(_N_ENEMY_BULLETS, dtype=jnp.float32),
            enemy_bullet_active=jnp.zeros(_N_ENEMY_BULLETS, dtype=jnp.bool_),
            wave=jnp.int32(0),
            fire_timer=jnp.int32(_FIRE_INTERVAL),
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: PhoenixState, action: jax.Array) -> PhoenixState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : PhoenixState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–7).

        Returns
        -------
        new_state : PhoenixState
            State after one emulated frame.
        """
        key, subkey = jax.random.split(state.key)

        # --- Action decode ---
        move_right = (action == 2) | (action == 5)
        move_left = (action == 3) | (action == 6)
        has_fire = (action == 1) | (action == 5) | (action == 6) | (action == 7)
        has_shield = (action == 4) | (action == 7)

        # --- Player movement ---
        dx = jnp.where(
            move_right,
            jnp.float32(_PLAYER_SPEED),
            jnp.where(move_left, jnp.float32(-_PLAYER_SPEED), jnp.float32(0.0)),
        )
        player_x = jnp.clip(state.player_x + dx, _PLAYER_LEFT, _PLAYER_RIGHT)
        shield_active = has_shield

        # --- Player fires ---
        fire = has_fire & ~state.bullet_active
        new_bx = jnp.where(fire, player_x + jnp.float32(_PLAYER_W / 2), state.bullet_x)
        new_by = jnp.where(fire, jnp.float32(_PLAYER_Y - _BULLET_H), state.bullet_y)
        bullet_active = state.bullet_active | fire

        # --- Move player bullet up ---
        new_by = jnp.where(bullet_active, new_by - jnp.float32(_BULLET_SPEED), new_by)
        pb_oob = bullet_active & (new_by < jnp.float32(_PLAY_TOP))
        bullet_active = bullet_active & ~pb_oob

        # --- Move enemy bullets down ---
        new_eby = jnp.where(
            state.enemy_bullet_active,
            state.enemy_bullet_y + jnp.float32(_ENEMY_BULLET_SPEED),
            state.enemy_bullet_y,
        )
        eb_oob = state.enemy_bullet_active & (
            new_eby + jnp.float32(_BULLET_H) > jnp.float32(_PLAYER_Y + _PLAYER_H)
        )
        enemy_bullet_active = state.enemy_bullet_active & ~eb_oob

        # --- Shield blocks enemy bullets ---
        shield_block = (
            enemy_bullet_active
            & shield_active
            & (new_eby + jnp.float32(_BULLET_H) > jnp.float32(_PLAYER_Y - 8))
            & (new_eby < jnp.float32(_PLAYER_Y + _PLAYER_H))
            & (state.enemy_bullet_x + jnp.float32(_BULLET_W) > player_x)
            & (state.enemy_bullet_x < player_x + jnp.float32(_PLAYER_W))
        )
        enemy_bullet_active = enemy_bullet_active & ~shield_block

        # --- Enemy bullet hits player (unshielded) ---
        ebx = state.enemy_bullet_x
        eb_hit = (
            enemy_bullet_active
            & ~shield_active
            & (ebx + jnp.float32(_BULLET_W) > player_x)
            & (ebx < player_x + jnp.float32(_PLAYER_W))
            & (new_eby + jnp.float32(_BULLET_H) > jnp.float32(_PLAYER_Y))
            & (new_eby < jnp.float32(_PLAYER_Y + _PLAYER_H))
        )
        lives_lost_by_bullet = jnp.minimum(
            jnp.sum(eb_hit.astype(jnp.int32)), jnp.int32(1)
        )
        enemy_bullet_active = enemy_bullet_active & ~eb_hit

        # --- Bird oscillation and descent ---
        wave_f = state.wave.astype(jnp.float32)
        descent_speed = jnp.float32(_BIRD_DESCENT) * (
            jnp.float32(1.0) + jnp.float32(0.2) * wave_f
        )

        new_bx_birds = state.bird_x + state.bird_dx
        # Bounce off walls
        b_hit_left = new_bx_birds < jnp.float32(0.0)
        b_hit_right = new_bx_birds + jnp.float32(_BIRD_W) > jnp.float32(160.0)
        new_bird_dx = jnp.where(b_hit_left | b_hit_right, -state.bird_dx, state.bird_dx)
        new_bx_birds = jnp.clip(
            new_bx_birds, jnp.float32(0.0), jnp.float32(160.0 - _BIRD_W)
        )
        new_bird_y = state.bird_y + descent_speed

        # --- Player bullet vs bird collision ---
        hit_mask = (
            state.bird_alive
            & bullet_active
            & (new_bx + jnp.float32(_BULLET_W) > new_bx_birds)
            & (new_bx < new_bx_birds + jnp.float32(_BIRD_W))
            & (new_by + jnp.float32(_BULLET_H) > new_bird_y)
            & (new_by < new_bird_y + jnp.float32(_BIRD_H))
        )
        any_hit = jnp.any(hit_mask)
        new_bird_alive = state.bird_alive & ~hit_mask
        step_reward = jnp.float32(
            jnp.sum((hit_mask * _BIRD_POINTS).astype(jnp.float32))
        )
        bullet_active = bullet_active & ~any_hit

        # --- Bird reaches player level → lose a life ---
        bird_at_player = new_bird_alive & (
            new_bird_y + jnp.float32(_BIRD_H) >= jnp.float32(_PLAYER_Y)
        )
        lives_lost_by_bird = jnp.minimum(
            jnp.sum(bird_at_player.astype(jnp.int32)), jnp.int32(1)
        )
        # Reset birds that reached the player back to top
        new_bird_y = jnp.where(bird_at_player, jnp.float32(_BIRD_Y0), new_bird_y)
        new_bird_alive = new_bird_alive  # birds don't die when reaching bottom

        # --- Wave clear → advance wave ---
        all_clear = ~jnp.any(new_bird_alive)
        new_wave = state.wave + jnp.where(all_clear, jnp.int32(1), jnp.int32(0))
        new_bird_alive = jnp.where(
            all_clear, jnp.ones(_N_BIRDS, dtype=jnp.bool_), new_bird_alive
        )
        new_bird_y = jnp.where(all_clear, _BIRD_INIT_Y.copy(), new_bird_y)
        new_bx_birds = jnp.where(all_clear, _BIRD_INIT_X.copy(), new_bx_birds)

        # --- Enemy fires ---
        n_alive = jnp.sum(new_bird_alive.astype(jnp.int32))
        rand_idx = jax.random.randint(subkey, (), 0, _N_BIRDS)
        fire_bird_alive = new_bird_alive[rand_idx]
        spawn_bx2 = new_bx_birds[rand_idx] + jnp.float32(_BIRD_W / 2)
        spawn_by2 = new_bird_y[rand_idx] + jnp.float32(_BIRD_H)

        should_fire = (
            (state.fire_timer <= jnp.int32(0))
            & fire_bird_alive
            & (n_alive > jnp.int32(0))
        )
        slot = state.step % jnp.int32(_N_ENEMY_BULLETS)
        slot_free = ~enemy_bullet_active[slot]
        do_fire = should_fire & slot_free

        new_ebx = state.enemy_bullet_x.at[slot].set(
            jnp.where(do_fire, spawn_bx2, state.enemy_bullet_x[slot])
        )
        new_eby_launch = new_eby.at[slot].set(
            jnp.where(do_fire, spawn_by2, new_eby[slot])
        )
        enemy_bullet_active = enemy_bullet_active.at[slot].set(
            enemy_bullet_active[slot] | do_fire
        )

        new_fire_timer = jnp.where(
            state.fire_timer <= jnp.int32(0),
            jnp.int32(_FIRE_INTERVAL),
            state.fire_timer - jnp.int32(1),
        )

        # --- Lives ---
        new_lives = state.lives - lives_lost_by_bullet - lives_lost_by_bird
        done = new_lives <= jnp.int32(0)

        return state.__replace__(
            player_x=player_x,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=bullet_active,
            shield_active=shield_active,
            bird_x=new_bx_birds,
            bird_y=new_bird_y,
            bird_dx=new_bird_dx,
            bird_alive=new_bird_alive,
            enemy_bullet_x=new_ebx,
            enemy_bullet_y=new_eby_launch,
            enemy_bullet_active=enemy_bullet_active,
            wave=new_wave,
            fire_timer=new_fire_timer,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            key=key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: PhoenixState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> PhoenixState:
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        return new_state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: PhoenixState) -> jax.Array:
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # --- Player ---
        px = jnp.int32(state.player_x)
        player_mask = (
            (_ROW_IDX_R >= _PLAYER_Y)
            & (_ROW_IDX_R < _PLAYER_Y + _PLAYER_H)
            & (_COL_IDX_R >= px)
            & (_COL_IDX_R < px + _PLAYER_W)
        )
        frame = jnp.where(player_mask[:, :, None], _PLAYER_COLOR[None, None, :], frame)

        # --- Shield (blue glow above player) ---
        shield_mask = (
            state.shield_active
            & (_ROW_IDX_R >= _PLAYER_Y - 6)
            & (_ROW_IDX_R < _PLAYER_Y)
            & (_COL_IDX_R >= px)
            & (_COL_IDX_R < px + _PLAYER_W)
        )
        frame = jnp.where(shield_mask[:, :, None], _SHIELD_COLOR[None, None, :], frame)

        # --- Player bullet ---
        bx = jnp.int32(state.bullet_x)
        by = jnp.int32(state.bullet_y)
        pb_mask = (
            state.bullet_active
            & (_ROW_IDX_R >= by)
            & (_ROW_IDX_R < by + _BULLET_H)
            & (_COL_IDX_R >= bx)
            & (_COL_IDX_R < bx + _BULLET_W)
        )
        frame = jnp.where(pb_mask[:, :, None], _BULLET_COLOR[None, None, :], frame)

        # --- Birds ---
        for i in range(_N_BIRDS):
            bxi = jnp.int32(state.bird_x[i])
            byi = jnp.int32(state.bird_y[i])
            color = jnp.where(_BIRD_ROW_IDX[i] < 2, _BIRD_TOP_COLOR, _BIRD_BOT_COLOR)
            bird_mask = (
                state.bird_alive[i]
                & (_ROW_IDX_R >= byi)
                & (_ROW_IDX_R < byi + _BIRD_H)
                & (_COL_IDX_R >= bxi)
                & (_COL_IDX_R < bxi + _BIRD_W)
            )
            frame = jnp.where(bird_mask[:, :, None], color[None, None, :], frame)

        # --- Enemy bullets ---
        for i in range(_N_ENEMY_BULLETS):
            ebx = jnp.int32(state.enemy_bullet_x[i])
            eby = jnp.int32(state.enemy_bullet_y[i])
            eb_mask = (
                state.enemy_bullet_active[i]
                & (_ROW_IDX_R >= eby)
                & (_ROW_IDX_R < eby + _BULLET_H)
                & (_COL_IDX_R >= ebx)
                & (_COL_IDX_R < ebx + _BULLET_W)
            )
            frame = jnp.where(
                eb_mask[:, :, None], _ENEMY_BULLET_COLOR[None, None, :], frame
            )

        return frame

    def _key_map(self):
        try:
            import pygame

            return {
                pygame.K_SPACE: 1,
                pygame.K_RIGHT: 2,
                pygame.K_d: 2,
                pygame.K_LEFT: 3,
                pygame.K_a: 3,
                pygame.K_LSHIFT: 4,
                pygame.K_RSHIFT: 4,
            }
        except ImportError:
            return {}
