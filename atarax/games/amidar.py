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

"""Amidar — JAX-native game implementation.

Paint every node of a 5×8 grid by walking to it while enemies patrol
the same paths.  Completing all four corners of a 2×2 block scores a
bonus.

Action space (5 actions):
    0 — NOOP
    1 — UP
    2 — RIGHT
    3 — DOWN
    4 — LEFT

Scoring:
    New cell painted — +1
    2×2 block complete — +50
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
_ROWS: int = 5  # node rows
_COLS: int = 8  # node cols
_N_NODES: int = _ROWS * _COLS  # 40

# Pixel positions for each node (y=row, x=col)
_NODE_X = jnp.array([8 + c * 20 for c in range(_COLS)], dtype=jnp.int32)  # [8]
_NODE_Y = jnp.array([20 + r * 30 for r in range(_ROWS)], dtype=jnp.int32)  # [5]

# Direction deltas indexed by action (0=NOOP, 1=UP, 2=RIGHT, 3=DOWN, 4=LEFT)
_ACT_DR = jnp.array([0, -1, 0, 1, 0], dtype=jnp.int32)
_ACT_DC = jnp.array([0, 0, 1, 0, -1], dtype=jnp.int32)

# Enemy count and starting positions (row, col)
_N_ENEMIES: int = 3
_ENEMY_STARTS = jnp.array([[0, 7], [4, 0], [2, 4]], dtype=jnp.int32)

# Player start
_PLAYER_START_R: int = 0
_PLAYER_START_C: int = 0

_MOVE_INTERVAL: int = 3
_ENEMY_INTERVAL: int = 5
_INIT_LIVES: int = 3

_PLAYER_W: int = 6
_ENEMY_W: int = 5

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([20, 20, 20], dtype=jnp.uint8)
_COLOR_GRID = jnp.array([80, 80, 80], dtype=jnp.uint8)
_COLOR_PAINTED = jnp.array([80, 180, 80], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 255, 80], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([220, 60, 60], dtype=jnp.uint8)


@chex.dataclass
class AmidarState(AtariState):
    """
    Complete Amidar game state — a JAX pytree.

    Parameters
    ----------
    player_row : jax.Array
        int32 — Player node row (0–4).
    player_col : jax.Array
        int32 — Player node column (0–7).
    painted : jax.Array
        bool[5, 8] — True where a node has been visited.
    enemy_row : jax.Array
        int32[3] — Enemy node rows.
    enemy_col : jax.Array
        int32[3] — Enemy node columns.
    enemy_dir : jax.Array
        int32[3] — Enemy movement direction (action index 1–4).
    move_timer : jax.Array
        int32 — Sub-steps until next player move.
    enemy_timer : jax.Array
        int32 — Sub-steps until next enemy step.
    key : jax.Array
        uint32[2] — PRNG for enemy direction randomisation.
    """

    player_row: jax.Array
    player_col: jax.Array
    painted: jax.Array
    enemy_row: jax.Array
    enemy_col: jax.Array
    enemy_dir: jax.Array
    move_timer: jax.Array
    enemy_timer: jax.Array
    key: jax.Array


class Amidar(AtariEnv):
    """
    Amidar implemented as a pure JAX function suite.

    Walk along a 5×8 grid painting nodes while avoiding enemies.
    Lives: 3.
    """

    num_actions: int = 5

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=28_000)

    def _reset(self, key: jax.Array) -> AmidarState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : AmidarState
            Player at (0,0), all cells unpainted, 3 lives.
        """
        painted = jnp.zeros((_ROWS, _COLS), dtype=jnp.bool_)
        painted = painted.at[_PLAYER_START_R, _PLAYER_START_C].set(True)
        return AmidarState(
            player_row=jnp.int32(_PLAYER_START_R),
            player_col=jnp.int32(_PLAYER_START_C),
            painted=painted,
            enemy_row=_ENEMY_STARTS[:, 0],
            enemy_col=_ENEMY_STARTS[:, 1],
            enemy_dir=jnp.array([2, 4, 3], dtype=jnp.int32),
            move_timer=jnp.int32(_MOVE_INTERVAL),
            enemy_timer=jnp.int32(_ENEMY_INTERVAL),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: AmidarState, action: jax.Array) -> AmidarState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : AmidarState
            Current game state.
        action : jax.Array
            int32 — Action index (0–4).

        Returns
        -------
        new_state : AmidarState
            State after one emulated frame.
        """
        key, sk = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Player movement
        new_move_timer = state.move_timer - jnp.int32(1)
        do_move = (new_move_timer <= jnp.int32(0)) & (action != jnp.int32(0))
        new_move_timer = jnp.where(do_move, jnp.int32(_MOVE_INTERVAL), new_move_timer)

        dr = _ACT_DR[action]
        dc = _ACT_DC[action]
        nr = jnp.clip(state.player_row + dr, 0, _ROWS - 1)
        nc = jnp.clip(state.player_col + dc, 0, _COLS - 1)
        # Only move if actually changed (no diagonal, no wall-clamp ghost move)
        actually_moved = (nr != state.player_row) | (nc != state.player_col)
        new_pr = jnp.where(do_move & actually_moved, nr, state.player_row)
        new_pc = jnp.where(do_move & actually_moved, nc, state.player_col)

        # Paint new cell
        was_painted = state.painted[new_pr, new_pc]
        new_painted = state.painted.at[new_pr, new_pc].set(True)
        step_reward = step_reward + jnp.where(
            was_painted, jnp.float32(0.0), jnp.float32(1.0)
        )

        # Box bonus: for each 2×2 block the player now completes, +50
        def check_block(carry, idx):
            r = idx // (_COLS - 1)
            c = idx % (_COLS - 1)
            # corners of block (r,c): (r,c), (r,c+1), (r+1,c), (r+1,c+1)
            all_painted = (
                new_painted[r, c]
                & new_painted[r, c + 1]
                & new_painted[r + 1, c]
                & new_painted[r + 1, c + 1]
            )
            prev_all = (
                state.painted[r, c]
                & state.painted[r, c + 1]
                & state.painted[r + 1, c]
                & state.painted[r + 1, c + 1]
            )
            newly_complete = all_painted & ~prev_all
            reward = carry + jnp.where(
                newly_complete, jnp.float32(50.0), jnp.float32(0.0)
            )
            return reward, None

        n_blocks = (_ROWS - 1) * (_COLS - 1)
        bonus_reward, _ = jax.lax.scan(
            check_block, jnp.float32(0.0), jnp.arange(n_blocks)
        )
        step_reward = step_reward + bonus_reward

        # Enemy movement
        new_enemy_timer = state.enemy_timer - jnp.int32(1)
        do_enemy = new_enemy_timer <= jnp.int32(0)
        new_enemy_timer = jnp.where(
            do_enemy, jnp.int32(_ENEMY_INTERVAL), new_enemy_timer
        )

        def move_enemy(i, carry):
            er, ec, edr, rng = carry
            rng, rng2 = jax.random.split(rng)
            cur_dir = edr[i]
            cur_dr = _ACT_DR[cur_dir]
            cur_dc = _ACT_DC[cur_dir]
            nr_ = er[i] + cur_dr
            nc_ = ec[i] + cur_dc
            # Check bounds; reverse or randomise at edges
            out_r = (nr_ < 0) | (nr_ >= _ROWS)
            out_c = (nc_ < 0) | (nc_ >= _COLS)
            out = out_r | out_c
            new_dir = jax.random.randint(rng2, (), 1, 5)  # 1–4
            cur_dir = jnp.where(out, new_dir, cur_dir)
            cur_dr = _ACT_DR[cur_dir]
            cur_dc = _ACT_DC[cur_dir]
            nr_ = jnp.clip(er[i] + cur_dr, 0, _ROWS - 1)
            nc_ = jnp.clip(ec[i] + cur_dc, 0, _COLS - 1)
            er = er.at[i].set(jnp.where(do_enemy, nr_, er[i]))
            ec = ec.at[i].set(jnp.where(do_enemy, nc_, ec[i]))
            edr = edr.at[i].set(jnp.where(do_enemy, cur_dir, edr[i]))
            return er, ec, edr, rng

        new_er, new_ec, new_edr, _ = jax.lax.fori_loop(
            0,
            _N_ENEMIES,
            move_enemy,
            (state.enemy_row, state.enemy_col, state.enemy_dir, sk),
        )

        # Enemy catches player
        caught = jnp.any((new_er == new_pr) & (new_ec == new_pc))
        new_lives = state.lives - jnp.where(caught, jnp.int32(1), jnp.int32(0))
        # Respawn on death
        new_pr = jnp.where(caught, jnp.int32(_PLAYER_START_R), new_pr)
        new_pc = jnp.where(caught, jnp.int32(_PLAYER_START_C), new_pc)
        new_er = jnp.where(caught, _ENEMY_STARTS[:, 0], new_er)
        new_ec = jnp.where(caught, _ENEMY_STARTS[:, 1], new_ec)

        all_painted = jnp.all(new_painted)
        done = (new_lives <= jnp.int32(0)) | all_painted

        return AmidarState(
            player_row=new_pr,
            player_col=new_pc,
            painted=new_painted,
            enemy_row=new_er,
            enemy_col=new_ec,
            enemy_dir=new_edr,
            move_timer=new_move_timer,
            enemy_timer=new_enemy_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: AmidarState, action: jax.Array) -> AmidarState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : AmidarState
            Current game state.
        action : jax.Array
            int32 — Action index (0–4).

        Returns
        -------
        new_state : AmidarState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: AmidarState) -> AmidarState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: AmidarState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : AmidarState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Draw horizontal grid lines (one per row of nodes)
        def draw_h_line(frm, r):
            y = _NODE_Y[r]
            x0 = _NODE_X[0]
            x1 = _NODE_X[_COLS - 1]
            mask = (_ROW_IDX == y) & (_COL_IDX >= x0) & (_COL_IDX <= x1)
            return jnp.where(mask[:, :, None], _COLOR_GRID, frm), None

        frame, _ = jax.lax.scan(draw_h_line, frame, jnp.arange(_ROWS))

        # Draw vertical grid lines (one per col of nodes)
        def draw_v_line(frm, c):
            x = _NODE_X[c]
            y0 = _NODE_Y[0]
            y1 = _NODE_Y[_ROWS - 1]
            mask = (_COL_IDX == x) & (_ROW_IDX >= y0) & (_ROW_IDX <= y1)
            return jnp.where(mask[:, :, None], _COLOR_GRID, frm), None

        frame, _ = jax.lax.scan(draw_v_line, frame, jnp.arange(_COLS))

        # Highlight painted nodes as filled squares
        def draw_painted(frm, idx):
            r = idx // _COLS
            c = idx % _COLS
            py = _NODE_Y[r]
            px = _NODE_X[c]
            mask = (
                state.painted[r, c]
                & (_ROW_IDX >= py - 3)
                & (_ROW_IDX <= py + 3)
                & (_COL_IDX >= px - 3)
                & (_COL_IDX <= px + 3)
            )
            return jnp.where(mask[:, :, None], _COLOR_PAINTED, frm), None

        frame, _ = jax.lax.scan(draw_painted, frame, jnp.arange(_N_NODES))

        # Draw enemies
        def draw_enemy(frm, i):
            py = _NODE_Y[state.enemy_row[i]]
            px = _NODE_X[state.enemy_col[i]]
            mask = (
                (_ROW_IDX >= py - _ENEMY_W // 2)
                & (_ROW_IDX <= py + _ENEMY_W // 2)
                & (_COL_IDX >= px - _ENEMY_W // 2)
                & (_COL_IDX <= px + _ENEMY_W // 2)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Draw player
        py = _NODE_Y[state.player_row]
        px = _NODE_X[state.player_col]
        player_mask = (
            (_ROW_IDX >= py - _PLAYER_W // 2)
            & (_ROW_IDX <= py + _PLAYER_W // 2)
            & (_COL_IDX >= px - _PLAYER_W // 2)
            & (_COL_IDX <= px + _PLAYER_W // 2)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Amidar action indices.
        """
        import pygame

        return {
            pygame.K_UP: 1,
            pygame.K_w: 1,
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_DOWN: 3,
            pygame.K_s: 3,
            pygame.K_LEFT: 4,
            pygame.K_a: 4,
        }
