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

"""Alien — JAX-native game implementation.

Navigate a space station armed with a flamethrower.  Destroy alien eggs
scattered around the corridors while avoiding contact with xenomorphs that
chase you through the rooms.  Shooting a xenomorph sends it back to its
spawn point.

Action space (6 actions):
    0 — NOOP
    1 — FIRE (shoot flamethrower)
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT

Scoring:
    Egg destroyed   — +10
    Alien shot      — +30
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------------
_TILE: int = 12  # pixels per tile
_COLS: int = 11  # grid columns
_ROWS: int = 14  # grid rows

# Pixel offsets for the maze area in the 160×210 frame
_MAZE_X0: int = (160 - _COLS * _TILE) // 2  # = (160 - 132) // 2 = 14
_MAZE_Y0: int = (210 - _ROWS * _TILE) // 2  # = (210 - 168) // 2 = 21

# Open arena — mostly passable with outer walls and a few internal walls
_WALL_RAW = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]
_WALL_ARR = jnp.array(_WALL_RAW, dtype=jnp.bool_)  # [14, 11]

# Egg positions: fixed tile positions (row, col)
_EGG_TILES = jnp.array(
    [[1, 1], [1, 9], [6, 1], [6, 9], [12, 1], [12, 9]], dtype=jnp.int32
)  # [6, 2]
_N_EGGS: int = 6

# Alien spawn positions and initial tiles
_N_ALIENS: int = 2
_ALIEN_STARTS = jnp.array([[2, 5], [11, 5]], dtype=jnp.int32)  # [2, 2] (row, col)

# Player start
_PLAYER_START = jnp.array([7, 5], dtype=jnp.int32)  # (row, col)

_MOVE_INTERVAL: int = 4  # sub-steps per tile move
_ALIEN_MOVE_INTERVAL: int = 6
_INIT_LIVES: int = 3

# Direction deltas: 0=NOOP, 1=UP, 2=RIGHT, 3=DOWN, 4=LEFT (action 1=FIRE, 2=UP, 3=RIGHT, 4=DOWN, 5=LEFT)
_ACT_DR = jnp.array([0, -1, 0, 1, 0, 0], dtype=jnp.int32)  # indexed by action
_ACT_DC = jnp.array([0, 0, 1, 0, -1, 0], dtype=jnp.int32)

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_WALL = jnp.array([40, 40, 120], dtype=jnp.uint8)
_COLOR_FLOOR = jnp.array([10, 10, 30], dtype=jnp.uint8)
_COLOR_EGG = jnp.array([0, 220, 0], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_ALIEN = jnp.array([200, 50, 50], dtype=jnp.uint8)
_COLOR_FLAME = jnp.array([255, 140, 0], dtype=jnp.uint8)


@chex.dataclass
class AlienState(AtariState):
    """
    Complete Alien game state — a JAX pytree.

    Parameters
    ----------
    player_row : jax.Array
        int32 — Player tile row.
    player_col : jax.Array
        int32 — Player tile column.
    player_dir : jax.Array
        int32 — Facing direction (action index 1–5, default 3 = RIGHT).
    eggs : jax.Array
        bool[6] — Surviving eggs.
    alien_row : jax.Array
        int32[2] — Alien tile rows.
    alien_col : jax.Array
        int32[2] — Alien tile columns.
    move_timer : jax.Array
        int32 — Sub-steps until next player step.
    alien_timer : jax.Array
        int32 — Sub-steps until next alien step.
    flame_active : jax.Array
        bool — Flame sprite visible.
    flame_timer : jax.Array
        int32 — Sub-steps flame remains visible.
    """

    player_row: jax.Array
    player_col: jax.Array
    player_dir: jax.Array
    eggs: jax.Array
    alien_row: jax.Array
    alien_col: jax.Array
    move_timer: jax.Array
    alien_timer: jax.Array
    flame_active: jax.Array
    flame_timer: jax.Array


class Alien(AtariEnv):
    """
    Alien implemented as a pure JAX function suite.

    Destroy all eggs without being caught by xenomorphs.  Lives: 3.
    """

    num_actions: int = 6

    def _reset(self, key: jax.Array) -> AlienState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : AlienState
            Player at start, all eggs present, 3 lives.
        """
        return AlienState(
            player_row=_PLAYER_START[0],
            player_col=_PLAYER_START[1],
            player_dir=jnp.int32(3),
            eggs=jnp.ones(_N_EGGS, dtype=jnp.bool_),
            alien_row=_ALIEN_STARTS[:, 0],
            alien_col=_ALIEN_STARTS[:, 1],
            move_timer=jnp.int32(_MOVE_INTERVAL),
            alien_timer=jnp.int32(_ALIEN_MOVE_INTERVAL),
            flame_active=jnp.bool_(False),
            flame_timer=jnp.int32(0),
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _try_move(self, row: jax.Array, col: jax.Array, dr: jax.Array, dc: jax.Array):
        """
        Attempt one-tile move; block if wall.

        Parameters
        ----------
        row : jax.Array
            int32 — Current row.
        col : jax.Array
            int32 — Current column.
        dr : jax.Array
            int32 — Row delta.
        dc : jax.Array
            int32 — Column delta.

        Returns
        -------
        new_row : jax.Array
            int32
        new_col : jax.Array
            int32
        """
        nr = jnp.clip(row + dr, 0, _ROWS - 1)
        nc = jnp.clip(col + dc, 0, _COLS - 1)
        blocked = _WALL_ARR[nr, nc]
        return jnp.where(blocked, row, nr), jnp.where(blocked, col, nc)

    def _step_physics(self, state: AlienState, action: jax.Array) -> AlienState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : AlienState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : AlienState
            State after one emulated frame.
        """
        step_reward = jnp.float32(0.0)

        # Player direction: fire keeps facing, movement updates dir
        is_move = (action >= jnp.int32(2)) & (action <= jnp.int32(5))
        new_dir = jnp.where(is_move, action, state.player_dir)

        # Player move (tick-based)
        new_move_timer = state.move_timer - jnp.int32(1)
        do_move = (new_move_timer <= jnp.int32(0)) & is_move
        new_move_timer = jnp.where(do_move, jnp.int32(_MOVE_INTERVAL), new_move_timer)

        dr = _ACT_DR[action]
        dc = _ACT_DC[action]
        new_pr, new_pc = self._try_move(state.player_row, state.player_col, dr, dc)
        new_pr = jnp.where(do_move, new_pr, state.player_row)
        new_pc = jnp.where(do_move, new_pc, state.player_col)

        # Fire flame in facing direction
        fire = action == jnp.int32(1)
        fdr = _ACT_DR[state.player_dir]
        fdc = _ACT_DC[state.player_dir]
        flame_r = new_pr + fdr
        flame_c = new_pc + fdc
        flame_valid = (
            fire
            & (flame_r >= 0)
            & (flame_r < _ROWS)
            & (flame_c >= 0)
            & (flame_c < _COLS)
            & ~_WALL_ARR[
                jnp.clip(flame_r, 0, _ROWS - 1), jnp.clip(flame_c, 0, _COLS - 1)
            ]
        )
        new_flame_active = jnp.where(flame_valid, jnp.bool_(True), state.flame_active)
        new_flame_timer = jnp.where(flame_valid, jnp.int32(3), state.flame_timer)
        new_flame_timer = jnp.where(
            new_flame_timer > jnp.int32(0), new_flame_timer - jnp.int32(1), jnp.int32(0)
        )
        new_flame_active = new_flame_active & (new_flame_timer > jnp.int32(0))

        # Flame hits eggs
        egg_rows = _EGG_TILES[:, 0]  # [6]
        egg_cols = _EGG_TILES[:, 1]  # [6]
        flame_hits_egg = (
            new_flame_active
            & (egg_rows == flame_r)
            & (egg_cols == flame_c)
            & state.eggs
        )
        newly_destroyed = flame_hits_egg
        new_eggs = state.eggs & ~newly_destroyed
        n_destroyed = jnp.sum(newly_destroyed).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_destroyed * 10)

        # Flame hits aliens
        alien_hit = (
            new_flame_active
            & (state.alien_row == flame_r)
            & (state.alien_col == flame_c)
        )
        n_alien_hit = jnp.sum(alien_hit).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_alien_hit * 30)
        # Respawn hit aliens at their start position
        new_alien_row = jnp.where(alien_hit, _ALIEN_STARTS[:, 0], state.alien_row)
        new_alien_col = jnp.where(alien_hit, _ALIEN_STARTS[:, 1], state.alien_col)

        # Alien movement (chase player)
        new_alien_timer = state.alien_timer - jnp.int32(1)
        do_alien_move = new_alien_timer <= jnp.int32(0)
        new_alien_timer = jnp.where(
            do_alien_move, jnp.int32(_ALIEN_MOVE_INTERVAL), new_alien_timer
        )

        def chase_one(i, carry):
            ar, ac = carry
            dr_ = jnp.sign(new_pr - ar[i]).astype(jnp.int32)
            dc_ = jnp.sign(new_pc - ac[i]).astype(jnp.int32)
            # Prefer horizontal movement
            nr_, nc_ = self._try_move(ar[i], ac[i], jnp.int32(0), dc_)
            blocked_h = (nr_ == ar[i]) & (nc_ == ac[i])
            nr_, nc_ = jnp.where(
                blocked_h,
                self._try_move(ar[i], ac[i], dr_, jnp.int32(0)),
                (nr_, nc_),
            )
            ar = ar.at[i].set(jnp.where(do_alien_move, nr_, ar[i]))
            ac = ac.at[i].set(jnp.where(do_alien_move, nc_, ac[i]))
            return ar, ac

        new_alien_row, new_alien_col = jax.lax.fori_loop(
            0, _N_ALIENS, chase_one, (new_alien_row, new_alien_col)
        )

        # Alien catches player
        alien_catches = (new_alien_row == new_pr) & (new_alien_col == new_pc)
        any_catch = jnp.any(alien_catches)
        new_lives = state.lives - jnp.where(any_catch, jnp.int32(1), jnp.int32(0))

        # Respawn player + aliens on death
        new_pr = jnp.where(any_catch, _PLAYER_START[0], new_pr)
        new_pc = jnp.where(any_catch, _PLAYER_START[1], new_pc)
        new_alien_row = jnp.where(any_catch, _ALIEN_STARTS[:, 0], new_alien_row)
        new_alien_col = jnp.where(any_catch, _ALIEN_STARTS[:, 1], new_alien_col)

        done = (new_lives <= jnp.int32(0)) | (~jnp.any(new_eggs))

        return AlienState(
            player_row=new_pr,
            player_col=new_pc,
            player_dir=new_dir,
            eggs=new_eggs,
            alien_row=new_alien_row,
            alien_col=new_alien_col,
            move_timer=new_move_timer,
            alien_timer=new_alien_timer,
            flame_active=new_flame_active,
            flame_timer=new_flame_timer,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=state.key,
        )

    def _step(self, state: AlienState, action: jax.Array) -> AlienState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : AlienState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : AlienState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: AlienState) -> AlienState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: AlienState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : AlienState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        # Maze background
        wall_pixels = jnp.repeat(jnp.repeat(_WALL_ARR, _TILE, axis=0), _TILE, axis=1)
        frame = jnp.where(
            wall_pixels[:, :, None],
            _COLOR_WALL,
            jnp.full((_ROWS * _TILE, _COLS * _TILE, 3), 0, dtype=jnp.uint8),
        )
        # Second pass: fill non-wall tiles with floor colour
        frame = jnp.where(~wall_pixels[:, :, None], _COLOR_FLOOR, frame)

        # Embed maze in 210×160 frame
        full = jnp.zeros((210, 160, 3), dtype=jnp.uint8)
        full = full.at[
            _MAZE_Y0 : _MAZE_Y0 + _ROWS * _TILE,
            _MAZE_X0 : _MAZE_X0 + _COLS * _TILE,
        ].set(frame)
        frame = full

        # Helper: draw a filled tile at (tile_row, tile_col)
        def draw_tile(frm, tile_row, tile_col, color):
            py = _MAZE_Y0 + tile_row * _TILE
            px = _MAZE_X0 + tile_col * _TILE
            mask = (
                (_ROW_IDX >= py)
                & (_ROW_IDX < py + _TILE)
                & (_COL_IDX >= px)
                & (_COL_IDX < px + _TILE)
            )
            return jnp.where(mask[:, :, None], color, frm)

        # Eggs
        def draw_egg(frm, i):
            er = _EGG_TILES[i, 0]
            ec = _EGG_TILES[i, 1]
            alive = state.eggs[i]
            py = _MAZE_Y0 + er * _TILE + 2
            px = _MAZE_X0 + ec * _TILE + 2
            mask = (
                alive
                & (_ROW_IDX >= py)
                & (_ROW_IDX < py + _TILE - 4)
                & (_COL_IDX >= px)
                & (_COL_IDX < px + _TILE - 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_EGG, frm), None

        frame, _ = jax.lax.scan(draw_egg, frame, jnp.arange(_N_EGGS))

        # Aliens
        def draw_alien(frm, i):
            ar = state.alien_row[i]
            ac = state.alien_col[i]
            return draw_tile(frm, ar, ac, _COLOR_ALIEN), None

        frame, _ = jax.lax.scan(draw_alien, frame, jnp.arange(_N_ALIENS))

        # Flame
        fdr = _ACT_DR[state.player_dir]
        fdc = _ACT_DC[state.player_dir]
        flame_r = state.player_row + fdr
        flame_c = state.player_col + fdc
        frame = jnp.where(
            state.flame_active,
            draw_tile(frame, flame_r, flame_c, _COLOR_FLAME),
            frame,
        )

        # Player
        frame = draw_tile(frame, state.player_row, state.player_col, _COLOR_PLAYER)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Alien action indices.
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
