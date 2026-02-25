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

"""Qbert — JAX-native game implementation.

Mechanics implemented directly in JAX with no hardware emulation.
All conditionals use `jnp.where`; the step loop uses `jax.lax.fori_loop`.

Pyramid layout (isometric, y=0 at top):
    6 rows (row 0 = apex, row 5 = base); valid cells: col ∈ [0, row].
    Qbert starts at the apex (row=0, col=0).
    Jump off-pyramid (col < 0, col > row, or row > 5) → life loss.

Action space (5 actions):
    0 — NOOP
    1 — UP-RIGHT   (row−1, col same)
    2 — UP-LEFT    (row−1, col−1)
    3 — DOWN-RIGHT (row+1, col+1)
    4 — DOWN-LEFT  (row+1, col same)

Scoring:
    +25 per newly coloured cube.  All 21 cubes coloured → episode ends (win).
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Pyramid geometry
# ---------------------------------------------------------------------------
_ROWS: int = 6  # rows 0–5 (row 0 = apex)
_MAX_CUBES: int = 21  # total valid cells

_CUBE_REWARD: float = 25.0

# Isometric screen geometry (pixels, y=0 at top)
_CUBE_W: int = 16  # width of a cube face
_CUBE_H: int = 10  # height of a cube face
_APEX_X: int = 80  # x-centre of apex cube
_APEX_Y: int = 40  # y-top of apex cube
_DX: int = 8  # x-offset between adjacent columns (half cube width)
_DY: int = 12  # y-offset between adjacent rows

# Player sprite size
_SPRITE_W: int = 6
_SPRITE_H: int = 8

# Enemy timing
_ENEMY_STEP_INTERVAL: int = 8  # sub-steps between enemy moves
_SPAWN_INTERVAL: int = 60  # sub-steps between enemy spawns
_JUMP_INTERVAL: int = 8  # sub-steps between player jumps

_INIT_LIVES: int = 3
_FRAME_SKIP: int = 4

# Precomputed cube centre positions (module-level concrete arrays)
# _CUBE_CX[row, col] = pixel x-centre of cube at (row, col); 0 where invalid
# _CUBE_CY[row, col] = pixel y-centre of cube at (row, col); 0 where invalid
_rows = jnp.arange(_ROWS)  # [6]
_cols = jnp.arange(_ROWS)  # [6]
_rr, _cc = jnp.meshgrid(_rows, _cols, indexing="ij")  # [6, 6] each
_valid = _cc <= _rr  # valid cells mask [6, 6]

# Centre x: apex at _APEX_X + offset from isometric projection
# For row r, col c: cx = _APEX_X + (c - r/2) * _CUBE_W
_CUBE_CX = jnp.where(
    _valid,
    jnp.int32(_APEX_X) + jnp.int32((_cc * _CUBE_W) - (_rr * _CUBE_W // 2)),
    jnp.int32(0),
)  # [6, 6]

_CUBE_CY = jnp.where(
    _valid,
    jnp.int32(_APEX_Y) + _rr * jnp.int32(_DY),
    jnp.int32(0),
)  # [6, 6]

# Precomputed index arrays for branch-free rendering
_ROW_IDX = jnp.arange(210)[:, None]  # [210, 1]
_COL_IDX = jnp.arange(160)[None, :]  # [1, 160]

# Cube colours: uncoloured (target) and coloured states
_COLOR_UNCOLOURED = jnp.array([80, 80, 200], dtype=jnp.uint8)  # blue-grey
_COLOR_COLOURED = jnp.array([200, 80, 80], dtype=jnp.uint8)  # red-orange
_COLOR_PLAYER = jnp.array([255, 200, 0], dtype=jnp.uint8)  # yellow
_COLOR_COILY = jnp.array([200, 50, 50], dtype=jnp.uint8)  # red
_COLOR_BALL = jnp.array([255, 140, 0], dtype=jnp.uint8)  # orange


@chex.dataclass
class QbertState(AtariState):
    """
    Complete Qbert game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score` from `AtariState`.

    Parameters
    ----------
    cubes : jax.Array
        bool[6, 6] — Cube coloured state.  `True` = target colour reached.
        Only cells where `col <= row` are valid.
    player_row : jax.Array
        int32 — Qbert's current row (0 = apex).
    player_col : jax.Array
        int32 — Qbert's current column.
    player_alive : jax.Array
        bool — `False` during respawn delay after life loss.
    coily_row : jax.Array
        int32 — Coily's row; -1 when inactive.
    coily_col : jax.Array
        int32 — Coily's column.
    coily_active : jax.Array
        bool — Whether Coily is on the board.
    red_ball_row : jax.Array
        int32 — Red ball row; -1 when inactive.
    red_ball_col : jax.Array
        int32 — Red ball column.
    red_ball_active : jax.Array
        bool — Whether the red ball is on the board.
    jump_timer : jax.Array
        int32 — Sub-steps until next player jump fires.
    enemy_timer : jax.Array
        int32 — Sub-steps until next enemy move.
    spawn_timer : jax.Array
        int32 — Sub-steps until next enemy spawn.
    key : jax.Array
        uint32[2] — PRNG key evolved each frame.
    """

    cubes: jax.Array
    player_row: jax.Array
    player_col: jax.Array
    player_alive: jax.Array
    coily_row: jax.Array
    coily_col: jax.Array
    coily_active: jax.Array
    red_ball_row: jax.Array
    red_ball_col: jax.Array
    red_ball_active: jax.Array
    jump_timer: jax.Array
    enemy_timer: jax.Array
    spawn_timer: jax.Array
    key: jax.Array


class Qbert(AtariEnv):
    """
    Qbert implemented as a pure JAX function suite.

    No hardware emulation — game physics are computed directly using
    `jnp.where` for all conditionals and `jax.lax.fori_loop` for the
    4-frame skip inside `_step`.

    The isometric pyramid is rendered from precomputed cube-centre arrays,
    drawing each cube's top-face diamond using branch-free masking.
    """

    num_actions: int = 5

    def _reset(self, key: jax.Array) -> QbertState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : QbertState
            All cubes uncoloured, Qbert at apex, no enemies, 3 lives.
        """
        return QbertState(
            cubes=jnp.zeros((_ROWS, _ROWS), dtype=jnp.bool_),
            player_row=jnp.int32(0),
            player_col=jnp.int32(0),
            player_alive=jnp.bool_(True),
            coily_row=jnp.int32(-1),
            coily_col=jnp.int32(-1),
            coily_active=jnp.bool_(False),
            red_ball_row=jnp.int32(-1),
            red_ball_col=jnp.int32(-1),
            red_ball_active=jnp.bool_(False),
            jump_timer=jnp.int32(_JUMP_INTERVAL),
            enemy_timer=jnp.int32(_ENEMY_STEP_INTERVAL),
            spawn_timer=jnp.int32(_SPAWN_INTERVAL),
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: QbertState, action: jax.Array) -> QbertState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : QbertState
            Current game state.
        action : jax.Array
            int32 — Action index (0=NOOP, 1=UP-RIGHT, 2=UP-LEFT,
            3=DOWN-RIGHT, 4=DOWN-LEFT).

        Returns
        -------
        new_state : QbertState
            State after one emulated frame.
        """
        key, subkey = jax.random.split(state.key)

        # --- Player jump (fires when jump_timer hits 0) ---
        should_jump = (state.jump_timer <= jnp.int32(0)) & state.player_alive

        # Map each action to (d_row, d_col) offsets
        act_dr = jnp.array([0, -1, -1, 1, 1], dtype=jnp.int32)
        act_dc = jnp.array([0, 0, -1, 1, 0], dtype=jnp.int32)
        d_row = jnp.where(should_jump, act_dr[action], jnp.int32(0))
        d_col = jnp.where(should_jump, act_dc[action], jnp.int32(0))

        new_player_row = state.player_row + d_row
        new_player_col = state.player_col + d_col

        # Check fall-off: invalid pyramid cell
        fell_off = should_jump & (
            (new_player_row < jnp.int32(0))
            | (new_player_row >= jnp.int32(_ROWS))
            | (new_player_col < jnp.int32(0))
            | (new_player_col > new_player_row)
        )

        # On fall-off: keep old position and trigger life loss
        new_player_row = jnp.where(fell_off, state.player_row, new_player_row)
        new_player_col = jnp.where(fell_off, state.player_col, new_player_col)

        # Colour cube at new position (only when player moved onto valid cell)
        on_valid = (
            should_jump
            & ~fell_off
            & (new_player_row >= jnp.int32(0))
            & (new_player_row < jnp.int32(_ROWS))
            & (new_player_col >= jnp.int32(0))
            & (new_player_col <= new_player_row)
        )
        was_uncoloured = ~state.cubes[new_player_row, new_player_col]
        cube_newly_coloured = on_valid & was_uncoloured
        step_reward = jnp.where(
            cube_newly_coloured, jnp.float32(_CUBE_REWARD), jnp.float32(0.0)
        )

        # Update cube grid functionally (safe even when on_valid=False)
        new_cubes = jnp.where(
            on_valid,
            state.cubes.at[new_player_row, new_player_col].set(jnp.bool_(True)),
            state.cubes,
        )

        # Jump timer reset
        new_jump_timer = jnp.where(
            should_jump,
            jnp.int32(_JUMP_INTERVAL),
            state.jump_timer - jnp.int32(1),
        )

        # --- Enemy movement ---
        should_enemy_move = state.enemy_timer <= jnp.int32(0)

        # Coily: move toward player (down-right or down-left)
        coily_dr = jnp.where(
            new_player_col >= state.coily_col, jnp.int32(1), jnp.int32(1)
        )
        coily_dc = jnp.where(
            new_player_col >= state.coily_col, jnp.int32(1), jnp.int32(0)
        )
        new_coily_row = jnp.where(
            should_enemy_move & state.coily_active,
            state.coily_row + coily_dr,
            state.coily_row,
        )
        new_coily_col = jnp.where(
            should_enemy_move & state.coily_active,
            state.coily_col + coily_dc,
            state.coily_col,
        )
        # Deactivate Coily when it falls off the pyramid bottom
        coily_off = state.coily_active & (new_coily_row >= jnp.int32(_ROWS))
        new_coily_row = jnp.where(coily_off, jnp.int32(-1), new_coily_row)
        new_coily_col = jnp.where(coily_off, jnp.int32(-1), new_coily_col)

        # Red ball: always moves down-right
        new_rb_row = jnp.where(
            should_enemy_move & state.red_ball_active,
            state.red_ball_row + jnp.int32(1),
            state.red_ball_row,
        )
        new_rb_col = jnp.where(
            should_enemy_move & state.red_ball_active,
            state.red_ball_col + jnp.int32(1),
            state.red_ball_col,
        )
        # Deactivate red ball when it falls off the pyramid
        rb_off = new_rb_row >= jnp.int32(_ROWS)
        new_rb_active = state.red_ball_active & ~rb_off
        new_rb_row = jnp.where(rb_off, jnp.int32(-1), new_rb_row)
        new_rb_col = jnp.where(rb_off, jnp.int32(-1), new_rb_col)

        new_enemy_timer = jnp.where(
            should_enemy_move,
            jnp.int32(_ENEMY_STEP_INTERVAL),
            state.enemy_timer - jnp.int32(1),
        )

        # --- Enemy–player collision ---
        coily_hits = (
            state.coily_active
            & (new_coily_row == new_player_row)
            & (new_coily_col == new_player_col)
            & state.player_alive
        )
        rb_hits = (
            new_rb_active
            & (new_rb_row == new_player_row)
            & (new_rb_col == new_player_col)
            & state.player_alive
        )
        enemy_hit = coily_hits | rb_hits

        # Life loss on fall-off or enemy hit
        lose_life = fell_off | enemy_hit
        new_lives = state.lives - jnp.where(lose_life, jnp.int32(1), jnp.int32(0))
        player_alive = state.player_alive & ~lose_life

        # Reset player to apex after life loss
        new_player_row = jnp.where(lose_life, jnp.int32(0), new_player_row)
        new_player_col = jnp.where(lose_life, jnp.int32(0), new_player_col)
        player_alive = jnp.where(lose_life, jnp.bool_(True), player_alive)

        # Deactivate Coily on hit or fall-off
        new_coily_active = state.coily_active & ~coily_hits & ~coily_off
        new_coily_row = jnp.where(coily_hits, jnp.int32(-1), new_coily_row)
        new_coily_col = jnp.where(coily_hits, jnp.int32(-1), new_coily_col)

        # --- Spawn timer ---
        should_spawn = state.spawn_timer <= jnp.int32(0)
        use_coily = jax.random.uniform(subkey) > jnp.float32(0.5)

        # Spawn Coily at apex if not active
        spawn_coily = should_spawn & ~new_coily_active & use_coily
        new_coily_active = jnp.where(spawn_coily, jnp.bool_(True), new_coily_active)
        new_coily_row = jnp.where(spawn_coily, jnp.int32(0), new_coily_row)
        new_coily_col = jnp.where(spawn_coily, jnp.int32(0), new_coily_col)

        # Spawn red ball at apex if not active and Coily not spawning
        spawn_rb = should_spawn & ~new_rb_active & ~use_coily
        new_rb_active = jnp.where(spawn_rb, jnp.bool_(True), new_rb_active)
        new_rb_row = jnp.where(spawn_rb, jnp.int32(0), new_rb_row)
        new_rb_col = jnp.where(spawn_rb, jnp.int32(0), new_rb_col)

        new_spawn_timer = jnp.where(
            should_spawn,
            jnp.int32(_SPAWN_INTERVAL),
            state.spawn_timer - jnp.int32(1),
        )

        # --- Episode end ---
        # Count valid cells and check if all are coloured
        valid_mask = _cc <= _rr  # [6, 6] - module-level
        all_coloured = jnp.all(jnp.where(valid_mask, new_cubes, jnp.bool_(True)))
        done = all_coloured | (new_lives <= jnp.int32(0))

        return QbertState(
            cubes=new_cubes,
            player_row=new_player_row,
            player_col=new_player_col,
            player_alive=player_alive,
            coily_row=new_coily_row,
            coily_col=new_coily_col,
            coily_active=new_coily_active,
            red_ball_row=new_rb_row,
            red_ball_col=new_rb_col,
            red_ball_active=new_rb_active,
            jump_timer=new_jump_timer,
            enemy_timer=new_enemy_timer,
            spawn_timer=new_spawn_timer,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=key,
        )

    def _step(self, state: QbertState, action: jax.Array) -> QbertState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : QbertState
            Current game state.
        action : jax.Array
            int32 — Action index (0–4).

        Returns
        -------
        new_state : QbertState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: QbertState) -> QbertState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, _FRAME_SKIP, body, state)

    def render(self, state: QbertState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        The isometric pyramid is drawn by iterating over all 21 valid cube
        cells using their precomputed centre positions.

        Parameters
        ----------
        state : QbertState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # Draw all 21 valid cubes as flat diamonds
        # For each (r, c) with c <= r: draw a diamond centred at (_CUBE_CX[r,c], _CUBE_CY[r,c])
        # Diamond: |dx/(_CUBE_W/2)| + |dy/(_CUBE_H/2)| <= 1
        # We iterate using lax.scan over a flat index of 21 cells
        flat_rows = jnp.array(
            [r for r in range(_ROWS) for c in range(r + 1)], dtype=jnp.int32
        )  # [21]
        flat_cols = jnp.array(
            [c for r in range(_ROWS) for c in range(r + 1)], dtype=jnp.int32
        )  # [21]

        def draw_cube(carry, idx):
            frm = carry
            r = flat_rows[idx]
            c = flat_cols[idx]
            cx = _CUBE_CX[r, c]
            cy = _CUBE_CY[r, c]
            coloured = state.cubes[r, c]
            color = jnp.where(coloured, _COLOR_COLOURED, _COLOR_UNCOLOURED)
            # Diamond mask: use taxi-cab distance in scaled coordinates
            dx = _COL_IDX - cx  # [1, 160]
            dy = _ROW_IDX - cy  # [210, 1]
            half_w = jnp.int32(_CUBE_W // 2)
            half_h = jnp.int32(_CUBE_H // 2)
            in_diamond = (jnp.abs(dx) * half_h + jnp.abs(dy) * half_w) <= (
                half_w * half_h
            )
            frm = jnp.where(in_diamond[:, :, None], color, frm)
            return frm, None

        frame, _ = jax.lax.scan(draw_cube, frame, jnp.arange(_MAX_CUBES))

        # Draw player (yellow sprite) at their cube position
        pcx = _CUBE_CX[state.player_row, state.player_col]
        pcy = _CUBE_CY[state.player_row, state.player_col] - jnp.int32(_SPRITE_H)
        player_mask = (
            state.player_alive
            & (_ROW_IDX >= pcy)
            & (_ROW_IDX < pcy + _SPRITE_H)
            & (_COL_IDX >= pcx - _SPRITE_W // 2)
            & (_COL_IDX < pcx + _SPRITE_W // 2)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        # Draw Coily (red) when active
        ccx = jnp.where(
            state.coily_active,
            _CUBE_CX[
                jnp.clip(state.coily_row, 0, _ROWS - 1),
                jnp.clip(state.coily_col, 0, _ROWS - 1),
            ],
            jnp.int32(-100),
        )
        ccy = jnp.where(
            state.coily_active,
            _CUBE_CY[
                jnp.clip(state.coily_row, 0, _ROWS - 1),
                jnp.clip(state.coily_col, 0, _ROWS - 1),
            ]
            - jnp.int32(_SPRITE_H),
            jnp.int32(-100),
        )
        coily_mask = (
            state.coily_active
            & (_ROW_IDX >= ccy)
            & (_ROW_IDX < ccy + _SPRITE_H)
            & (_COL_IDX >= ccx - _SPRITE_W // 2)
            & (_COL_IDX < ccx + _SPRITE_W // 2)
        )
        frame = jnp.where(coily_mask[:, :, None], _COLOR_COILY, frame)

        # Draw red ball when active
        rbcx = jnp.where(
            state.red_ball_active,
            _CUBE_CX[
                jnp.clip(state.red_ball_row, 0, _ROWS - 1),
                jnp.clip(state.red_ball_col, 0, _ROWS - 1),
            ],
            jnp.int32(-100),
        )
        rbcy = jnp.where(
            state.red_ball_active,
            _CUBE_CY[
                jnp.clip(state.red_ball_row, 0, _ROWS - 1),
                jnp.clip(state.red_ball_col, 0, _ROWS - 1),
            ]
            - jnp.int32(_SPRITE_H // 2),
            jnp.int32(-100),
        )
        rb_mask = (
            state.red_ball_active
            & (_ROW_IDX >= rbcy)
            & (_ROW_IDX < rbcy + _SPRITE_H // 2)
            & (_COL_IDX >= rbcx - _SPRITE_W // 2)
            & (_COL_IDX < rbcx + _SPRITE_W // 2)
        )
        frame = jnp.where(rb_mask[:, :, None], _COLOR_BALL, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Qbert action indices.
            Actions: 0=NOOP, 1=UP-RIGHT, 2=UP-LEFT, 3=DOWN-RIGHT, 4=DOWN-LEFT.
        """
        import pygame

        return {
            pygame.K_UP: 1,
            pygame.K_RIGHT: 1,
            pygame.K_LEFT: 2,
            pygame.K_DOWN: 4,
        }
