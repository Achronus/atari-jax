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

"""Ms. Pac-Man — JAX-native game implementation.

Mechanics implemented directly in JAX with no hardware emulation.
All conditionals use `jnp.where`; the step loop uses `jax.lax.fori_loop`.

Maze: 28×31 tiles, each tile 5×5 pixels.  Tile (0,0) = top-left.
The maze is stored as a module-level `bool[31, 28]` wall array.
Characters move one tile per agent step (after 4× frame-skip).

Action space (5 actions):
    0 — NOOP   (continue current direction)
    1 — UP
    2 — DOWN
    3 — LEFT
    4 — RIGHT

Scoring:
    Dot          — 10 pts
    Power pellet — 50 pts
    Ghost (chain)— 200 → 400 → 800 → 1600 pts
    Fruit        — 100 pts
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Maze constants
# ---------------------------------------------------------------------------
_TILE: int = 5  # pixels per tile
_COLS: int = 28  # maze width in tiles
_ROWS: int = 31  # maze height in tiles

# Pixel dimensions of the maze area
_MAZE_X0: int = 0  # left edge of tile (0, 0) in frame
_MAZE_Y0: int = 0  # top edge of tile (0, 0) in frame

_INIT_LIVES: int = 3
_FRAME_SKIP: int = 4

# Ghost frightened duration (agent steps)
_FRIGHTENED_DURATION: int = 30

# Fruit appearance thresholds (dots eaten)
_FRUIT_THRESHOLD: int = 70
_FRUIT_DURATION: int = 20  # agent steps until fruit despawns

# Ghost chain values (index = chain position 0–3)
_GHOST_CHAIN = jnp.array([200, 400, 800, 1600], dtype=jnp.float32)

# ---------------------------------------------------------------------------
# Level 1 maze — True = wall, False = passable
# 28 columns × 31 rows; standard Pac-Man layout (simplified).
# Dots are placed on all passable non-starting positions.
# ---------------------------------------------------------------------------
# fmt: off
_WALL_RAW = [
    # col: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # row 0
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 1
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],  # row 2
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],  # row 3
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],  # row 4
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 5
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1],  # row 6 (adjusted)
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1],  # row 7
    [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],  # row 8
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # row 9
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # row 10
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # row 11
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # row 12
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # row 13
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # row 14
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # row 15
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # row 16
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # row 17
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # row 18
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # row 19
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 20
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],  # row 21
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],  # row 22
    [1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1],  # row 23 (3=power pellet)
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # row 24
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # row 25
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 26
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],  # row 27
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],  # row 28
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # row 29
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # row 30
]
# fmt: on

# Build wall and dot arrays at module level (concrete JAX arrays)
_WALL_ARR = jnp.array(
    [[1 if v in (1,) else 0 for v in row] for row in _WALL_RAW],
    dtype=jnp.bool_,
)  # True = wall

# Passable tiles (not wall)
_PASSABLE = ~_WALL_ARR  # bool[31, 28]

# Initial dots: passable tiles that aren't the Pac-Man start (row 23, col 13-14)
# or ghost house area (rows 12-16, cols 11-16)
_PACMAN_START_ROW: int = 23
_PACMAN_START_COL: int = 13
_GHOST_HOUSE_ROWS = slice(12, 17)
_GHOST_HOUSE_COLS = slice(11, 17)

_dot_mask = jnp.array(
    [[1 if v == 0 else 0 for v in row] for row in _WALL_RAW],
    dtype=jnp.bool_,
)

# Power pellet positions (encoded as 3 in _WALL_RAW)
_pellet_mask = jnp.array(
    [[1 if v == 3 else 0 for v in row] for row in _WALL_RAW],
    dtype=jnp.bool_,
)

# Remove power pellets from dot mask (they're tracked separately)
_DOT_INIT = _dot_mask & ~_pellet_mask  # bool[31, 28]
_PELLET_INIT = _pellet_mask  # bool[31, 28]

# Number of initial dots
_N_DOTS: int = int(_DOT_INIT.sum())
_N_PELLETS: int = int(_PELLET_INIT.sum())

# Ghost starting positions (tile coords)
_GHOST_STARTS = jnp.array(
    [[13, 11], [14, 11], [13, 13], [14, 13]], dtype=jnp.int32
)  # [4, 2] — (col, row) pairs

# Pac-Man start
_PM_START_COL: int = 13
_PM_START_ROW: int = 23

# Fruit position (centre of maze)
_FRUIT_COL: int = 13
_FRUIT_ROW: int = 17

# Direction vectors: 0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
_DIR_DC = jnp.array([0, 0, 0, -1, 1], dtype=jnp.int32)
_DIR_DR = jnp.array([0, -1, 1, 0, 0], dtype=jnp.int32)

# Precomputed index arrays for branch-free rendering
_ROW_IDX = jnp.arange(210)[:, None]  # [210, 1]
_COL_IDX = jnp.arange(160)[None, :]  # [1, 160]

# Colours
_COLOR_WALL = jnp.array([0, 0, 200], dtype=jnp.uint8)
_COLOR_DOT = jnp.array([255, 220, 180], dtype=jnp.uint8)
_COLOR_PELLET = jnp.array([255, 220, 180], dtype=jnp.uint8)
_COLOR_PACMAN = jnp.array([255, 255, 0], dtype=jnp.uint8)
_COLOR_FRUIT = jnp.array([255, 0, 128], dtype=jnp.uint8)
_GHOST_COLORS = jnp.array(
    [[255, 0, 0], [255, 182, 255], [0, 255, 255], [255, 182, 82]],
    dtype=jnp.uint8,
)  # Blinky, Pinky, Inky, Sue
_COLOR_FRIGHTENED = jnp.array([0, 0, 255], dtype=jnp.uint8)


@chex.dataclass
class MsPacmanState(AtariState):
    """
    Complete Ms. Pac-Man game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score` from `AtariState`.

    Parameters
    ----------
    dots : jax.Array
        bool[31, 28] — Remaining dots.  `True` = dot present.
    pellets : jax.Array
        bool[31, 28] — Remaining power pellets.
    player_col : jax.Array
        int32 — Pac-Man tile column.
    player_row : jax.Array
        int32 — Pac-Man tile row.
    player_dir : jax.Array
        int32 — Current direction (0=NOOP/stopped, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT).
    ghost_col : jax.Array
        int32[4] — Ghost tile columns.
    ghost_row : jax.Array
        int32[4] — Ghost tile rows.
    ghost_dir : jax.Array
        int32[4] — Ghost directions.
    ghost_frightened : jax.Array
        bool[4] — Frightened mode per ghost.
    frightened_timer : jax.Array
        int32 — Agent steps remaining in power mode.
    ghost_eaten_count : jax.Array
        int32 — Ghosts eaten in current power pellet chain.
    fruit_active : jax.Array
        bool — Whether fruit is on screen.
    fruit_timer : jax.Array
        int32 — Steps until fruit despawns.
    dots_eaten : jax.Array
        int32 — Total dots eaten this episode (triggers fruit appearance).
    key : jax.Array
        uint32[2] — PRNG key for ghost AI junction decisions.
    """

    dots: jax.Array
    pellets: jax.Array
    player_col: jax.Array
    player_row: jax.Array
    player_dir: jax.Array
    ghost_col: jax.Array
    ghost_row: jax.Array
    ghost_dir: jax.Array
    ghost_frightened: jax.Array
    frightened_timer: jax.Array
    ghost_eaten_count: jax.Array
    fruit_active: jax.Array
    fruit_timer: jax.Array
    dots_eaten: jax.Array
    key: jax.Array


class MsPacman(AtariEnv):
    """
    Ms. Pac-Man implemented as a pure JAX function suite.

    No hardware emulation — game physics are computed directly using
    `jnp.where` for all conditionals and `jax.lax.fori_loop` for the
    4-frame skip inside `_step`.

    Movement is tile-based: characters move one tile per agent step.
    The maze is a fixed 28×31 grid stored as a module-level constant.
    """

    num_actions: int = 5

    def _reset(self, key: jax.Array) -> MsPacmanState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : MsPacmanState
            Full dot grid, Pac-Man at start, ghosts in house, 3 lives.
        """
        return MsPacmanState(
            dots=_DOT_INIT,
            pellets=_PELLET_INIT,
            player_col=jnp.int32(_PM_START_COL),
            player_row=jnp.int32(_PM_START_ROW),
            player_dir=jnp.int32(0),
            ghost_col=_GHOST_STARTS[:, 0],
            ghost_row=_GHOST_STARTS[:, 1],
            ghost_dir=jnp.ones(4, dtype=jnp.int32),  # all start heading up
            ghost_frightened=jnp.zeros(4, dtype=jnp.bool_),
            frightened_timer=jnp.int32(0),
            ghost_eaten_count=jnp.int32(0),
            fruit_active=jnp.bool_(False),
            fruit_timer=jnp.int32(0),
            dots_eaten=jnp.int32(0),
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _try_move(self, col: jax.Array, row: jax.Array, direction: jax.Array):
        """
        Attempt to move one tile in `direction`; stay if wall blocks.

        Parameters
        ----------
        col : jax.Array
            int32 — Current tile column.
        row : jax.Array
            int32 — Current tile row.
        direction : jax.Array
            int32 — Direction index (0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT).

        Returns
        -------
        new_col : jax.Array
            int32 — New column after move attempt.
        new_row : jax.Array
            int32 — New row after move attempt.
        moved : jax.Array
            bool — `True` if move succeeded (destination not a wall).
        """
        dc = _DIR_DC[direction]
        dr = _DIR_DR[direction]
        nc = jnp.clip(col + dc, 0, _COLS - 1)
        nr = jnp.clip(row + dr, 0, _ROWS - 1)
        blocked = _WALL_ARR[nr, nc]
        new_col = jnp.where(blocked, col, nc)
        new_row = jnp.where(blocked, row, nr)
        moved = ~blocked & (direction != jnp.int32(0))
        return new_col, new_row, moved

    def _ghost_move(
        self,
        g_col: jax.Array,
        g_row: jax.Array,
        g_dir: jax.Array,
        p_col: jax.Array,
        p_row: jax.Array,
        frightened: jax.Array,
        key: jax.Array,
    ):
        """
        Move one ghost one tile, choosing direction at junctions stochastically.

        Parameters
        ----------
        g_col : jax.Array
            int32 — Ghost column.
        g_row : jax.Array
            int32 — Ghost row.
        g_dir : jax.Array
            int32 — Ghost's current direction.
        p_col : jax.Array
            int32 — Pac-Man column (for chase targeting).
        p_row : jax.Array
            int32 — Pac-Man row.
        frightened : jax.Array
            bool — Whether this ghost is frightened.
        key : jax.Array
            uint32[2] — PRNG key for random junction choice.

        Returns
        -------
        new_col : jax.Array
            int32
        new_row : jax.Array
            int32
        new_dir : jax.Array
            int32
        """
        # Try moving in each of 4 cardinal directions (1=UP,2=DOWN,3=LEFT,4=RIGHT)
        dirs = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        # Reverse direction (can't reverse unless it's the only option)
        reverse = jnp.array([2, 1, 4, 3], dtype=jnp.int32)

        # Score each direction: toward player (chase) or away (frightened)
        # Use negative L1 distance as score for chase; positive for scatter
        dc = _DIR_DC[dirs]  # [4]
        dr = _DIR_DR[dirs]  # [4]
        nc = jnp.clip(g_col + dc, 0, _COLS - 1)  # [4]
        nr = jnp.clip(g_row + dr, 0, _ROWS - 1)  # [4]

        # Passable and not reversing
        passable = ~_WALL_ARR[nr, nc]  # bool[4] — broadcast over (nc,nr) pairs

        # Manhattan distance to player
        dist = jnp.abs(nc - p_col) + jnp.abs(nr - p_row)  # [4]

        # Prefer the direction that minimises (chase) or maximises (frightened) distance
        # Add random noise for frightened mode and tie-breaking
        noise = jax.random.uniform(key, shape=(4,))
        score = jnp.where(
            frightened,
            jnp.float32(10.0) * noise,
            -dist.astype(jnp.float32) + noise,
        )

        # Reverse direction is penalty-weighted unless it's the only option
        is_reverse = (dirs == reverse[g_dir - 1]) & (g_dir >= jnp.int32(1))
        score = jnp.where(
            is_reverse & passable.any(), score - jnp.float32(100.0), score
        )

        # Mask out walls (−inf)
        score = jnp.where(passable, score, jnp.float32(-1e9))

        # Pick best direction
        best_dir_idx = jnp.argmax(score)
        new_dir = dirs[best_dir_idx]
        new_col = nc[best_dir_idx]
        new_row = nr[best_dir_idx]

        return new_col, new_row, new_dir

    def _step_physics(self, state: MsPacmanState, action: jax.Array) -> MsPacmanState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : MsPacmanState
            Current game state.
        action : jax.Array
            int32 — Action index (0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT).

        Returns
        -------
        new_state : MsPacmanState
            State after one emulated frame.
        """
        key, k1, k2, k3, k4 = jax.random.split(state.key, 5)
        ghost_keys = jnp.stack([k1, k2, k3, k4])  # [4, 2]

        step_reward = jnp.float32(0.0)

        # --- Pac-Man movement ---
        # Try new direction first; if blocked, continue in current direction
        wants_dc = _DIR_DC[action]
        wants_dr = _DIR_DR[action]
        want_nc = jnp.clip(state.player_col + wants_dc, 0, _COLS - 1)
        want_nr = jnp.clip(state.player_row + wants_dr, 0, _ROWS - 1)
        want_blocked = _WALL_ARR[want_nr, want_nc] | (action == jnp.int32(0))

        # Fall back to current direction
        new_col, new_row, moved_current = self._try_move(
            state.player_col, state.player_row, state.player_dir
        )
        # Use requested direction if not blocked
        p_col = jnp.where(want_blocked, new_col, want_nc)
        p_row = jnp.where(want_blocked, new_row, want_nr)
        p_dir = jnp.where(want_blocked, state.player_dir, action)

        # --- Dot / pellet consumption ---
        ate_dot = state.dots[p_row, p_col]
        new_dots = jnp.where(
            ate_dot, state.dots.at[p_row, p_col].set(False), state.dots
        )
        step_reward = step_reward + jnp.where(
            ate_dot, jnp.float32(10.0), jnp.float32(0.0)
        )
        new_dots_eaten = state.dots_eaten + jnp.where(
            ate_dot, jnp.int32(1), jnp.int32(0)
        )

        ate_pellet = state.pellets[p_row, p_col]
        new_pellets = jnp.where(
            ate_pellet, state.pellets.at[p_row, p_col].set(False), state.pellets
        )
        step_reward = step_reward + jnp.where(
            ate_pellet, jnp.float32(50.0), jnp.float32(0.0)
        )
        new_frightened = jnp.where(
            ate_pellet, jnp.ones(4, dtype=jnp.bool_), state.ghost_frightened
        )
        new_frightened_timer = jnp.where(
            ate_pellet, jnp.int32(_FRIGHTENED_DURATION), state.frightened_timer
        )
        new_eaten_count = jnp.where(ate_pellet, jnp.int32(0), state.ghost_eaten_count)

        # --- Frightened timer tick ---
        new_frightened_timer = jnp.where(
            new_frightened_timer > jnp.int32(0),
            new_frightened_timer - jnp.int32(1),
            jnp.int32(0),
        )
        new_frightened = jnp.where(
            new_frightened_timer == jnp.int32(0),
            jnp.zeros(4, dtype=jnp.bool_),
            new_frightened,
        )

        # --- Ghost movement ---
        # Move each ghost using vectorised approach
        g_cols = state.ghost_col  # int32[4]
        g_rows = state.ghost_row  # int32[4]
        g_dirs = state.ghost_dir  # int32[4]

        # Use lax.fori_loop over 4 ghosts
        def move_ghost(i, carry):
            gcols, grows, gdirs = carry
            gc = gcols[i]
            gr = grows[i]
            gd = gdirs[i]
            gf = new_frightened[i]
            gk = ghost_keys[i]
            nc, nr, nd = self._ghost_move(gc, gr, gd, p_col, p_row, gf, gk)
            gcols = gcols.at[i].set(nc)
            grows = grows.at[i].set(nr)
            gdirs = gdirs.at[i].set(nd)
            return gcols, grows, gdirs

        g_cols, g_rows, g_dirs = jax.lax.fori_loop(
            0, 4, move_ghost, (g_cols, g_rows, g_dirs)
        )

        # --- Ghost–player collision ---
        at_player = (g_cols == p_col) & (g_rows == p_row)  # bool[4]
        eat_ghost = at_player & new_frightened  # bool[4]
        ghost_kills = at_player & ~new_frightened  # bool[4]

        # Eating ghosts
        n_eaten_this_step = jnp.sum(eat_ghost).astype(jnp.int32)
        chain_idx = jnp.clip(new_eaten_count, 0, 3)
        ghost_eat_reward = jnp.where(
            n_eaten_this_step > jnp.int32(0),
            _GHOST_CHAIN[chain_idx],
            jnp.float32(0.0),
        )
        step_reward = step_reward + ghost_eat_reward
        new_eaten_count = new_eaten_count + n_eaten_this_step
        # Reset eaten ghosts to house
        g_cols = jnp.where(eat_ghost, _GHOST_STARTS[:, 0], g_cols)
        g_rows = jnp.where(eat_ghost, _GHOST_STARTS[:, 1], g_rows)
        new_frightened = new_frightened & ~eat_ghost

        # Life loss
        any_kill = jnp.any(ghost_kills)
        new_lives = state.lives - jnp.where(any_kill, jnp.int32(1), jnp.int32(0))

        # Reset Pac-Man and ghosts on death
        p_col = jnp.where(any_kill, jnp.int32(_PM_START_COL), p_col)
        p_row = jnp.where(any_kill, jnp.int32(_PM_START_ROW), p_row)
        p_dir = jnp.where(any_kill, jnp.int32(0), p_dir)
        g_cols = jnp.where(any_kill, _GHOST_STARTS[:, 0], g_cols)
        g_rows = jnp.where(any_kill, _GHOST_STARTS[:, 1], g_rows)
        new_frightened = jnp.where(
            any_kill, jnp.zeros(4, dtype=jnp.bool_), new_frightened
        )

        # --- Fruit ---
        fruit_spawn = (
            ~state.fruit_active
            & (new_dots_eaten >= jnp.int32(_FRUIT_THRESHOLD))
            & (new_dots_eaten - jnp.int32(1) < jnp.int32(_FRUIT_THRESHOLD))
        )
        new_fruit_active = jnp.where(fruit_spawn, jnp.bool_(True), state.fruit_active)
        new_fruit_timer = jnp.where(
            fruit_spawn, jnp.int32(_FRUIT_DURATION), state.fruit_timer
        )

        ate_fruit = (
            new_fruit_active
            & (p_col == jnp.int32(_FRUIT_COL))
            & (p_row == jnp.int32(_FRUIT_ROW))
        )
        step_reward = step_reward + jnp.where(
            ate_fruit, jnp.float32(100.0), jnp.float32(0.0)
        )
        new_fruit_active = new_fruit_active & ~ate_fruit

        # Fruit despawn timer
        new_fruit_timer = jnp.where(
            new_fruit_active,
            new_fruit_timer - jnp.int32(1),
            jnp.int32(0),
        )
        fruit_expired = new_fruit_active & (new_fruit_timer <= jnp.int32(0))
        new_fruit_active = new_fruit_active & ~fruit_expired

        # --- Level clear: reset dots when all consumed ---
        all_dots_eaten = ~jnp.any(new_dots) & ~jnp.any(new_pellets)
        new_dots = jnp.where(all_dots_eaten, _DOT_INIT, new_dots)
        new_pellets = jnp.where(all_dots_eaten, _PELLET_INIT, new_pellets)
        new_dots_eaten = jnp.where(all_dots_eaten, jnp.int32(0), new_dots_eaten)

        # --- Episode end ---
        done = new_lives <= jnp.int32(0)

        return MsPacmanState(
            dots=new_dots,
            pellets=new_pellets,
            player_col=p_col,
            player_row=p_row,
            player_dir=p_dir,
            ghost_col=g_cols,
            ghost_row=g_rows,
            ghost_dir=g_dirs,
            ghost_frightened=new_frightened,
            frightened_timer=new_frightened_timer,
            ghost_eaten_count=new_eaten_count,
            fruit_active=new_fruit_active,
            fruit_timer=new_fruit_timer,
            dots_eaten=new_dots_eaten,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=key,
        )

    def _step(self, state: MsPacmanState, action: jax.Array) -> MsPacmanState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : MsPacmanState
            Current game state.
        action : jax.Array
            int32 — Action index (0–4).

        Returns
        -------
        new_state : MsPacmanState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: MsPacmanState) -> MsPacmanState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, _FRAME_SKIP, body, state)

    def render(self, state: MsPacmanState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : MsPacmanState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # --- Maze walls ---
        # Each tile (row, col) maps to pixel region [row*T:(row+1)*T, col*T:(col+1)*T]
        # Expand wall mask to pixel grid using repeat
        wall_pixels = jnp.repeat(jnp.repeat(_WALL_ARR, _TILE, axis=0), _TILE, axis=1)
        # wall_pixels: bool[155, 140]; embed at (0, 10) to centre in 210×160 frame
        maze_x0 = (160 - _COLS * _TILE) // 2  # = (160 - 140) // 2 = 10
        maze_y0 = (210 - _ROWS * _TILE) // 2  # = (210 - 155) // 2 = 27

        frame = frame.at[
            maze_y0 : maze_y0 + _ROWS * _TILE,
            maze_x0 : maze_x0 + _COLS * _TILE,
        ].set(
            jnp.where(
                wall_pixels[:, :, None],
                _COLOR_WALL,
                jnp.zeros((_ROWS * _TILE, _COLS * _TILE, 3), dtype=jnp.uint8),
            )
        )

        # --- Dots ---
        dot_pixels = jnp.repeat(jnp.repeat(state.dots, _TILE, axis=0), _TILE, axis=1)
        # Centre a 1×1 dot in the middle of each tile
        dot_centre = jnp.zeros((_ROWS * _TILE, _COLS * _TILE), dtype=jnp.bool_)
        ry = jnp.arange(_ROWS * _TILE)
        rx = jnp.arange(_COLS * _TILE)
        dot_centre_mask = (
            ((ry % _TILE == _TILE // 2)[:, None])
            & ((rx % _TILE == _TILE // 2)[None, :])
        )
        dot_draw = dot_pixels & dot_centre_mask  # [155, 140]
        frame = frame.at[
            maze_y0 : maze_y0 + _ROWS * _TILE,
            maze_x0 : maze_x0 + _COLS * _TILE,
        ].set(
            jnp.where(
                dot_draw[:, :, None],
                _COLOR_DOT,
                frame[
                    maze_y0 : maze_y0 + _ROWS * _TILE,
                    maze_x0 : maze_x0 + _COLS * _TILE,
                ],
            )
        )

        # --- Power pellets (3×3 centre) ---
        pellet_pixels = jnp.repeat(
            jnp.repeat(state.pellets, _TILE, axis=0), _TILE, axis=1
        )
        pellet_centre_mask = (
            (ry % _TILE >= _TILE // 2 - 1)[:, None]
            & (ry % _TILE <= _TILE // 2 + 1)[:, None]
            & ((rx % _TILE >= _TILE // 2 - 1)[None, :])
            & ((rx % _TILE <= _TILE // 2 + 1)[None, :])
        )
        pellet_draw = pellet_pixels & pellet_centre_mask
        frame = frame.at[
            maze_y0 : maze_y0 + _ROWS * _TILE,
            maze_x0 : maze_x0 + _COLS * _TILE,
        ].set(
            jnp.where(
                pellet_draw[:, :, None],
                _COLOR_PELLET,
                frame[
                    maze_y0 : maze_y0 + _ROWS * _TILE,
                    maze_x0 : maze_x0 + _COLS * _TILE,
                ],
            )
        )

        # Helper: draw a filled tile-sized rectangle at (tile_col, tile_row)
        def draw_tile(frm, tile_col, tile_row, color):
            py = maze_y0 + tile_row * _TILE
            px = maze_x0 + tile_col * _TILE
            mask = (
                (_ROW_IDX >= py)
                & (_ROW_IDX < py + _TILE)
                & (_COL_IDX >= px)
                & (_COL_IDX < px + _TILE)
            )
            return jnp.where(mask[:, :, None], color, frm)

        # --- Pac-Man ---
        frame = draw_tile(frame, state.player_col, state.player_row, _COLOR_PACMAN)

        # --- Ghosts ---
        def draw_ghost(carry, i):
            frm = carry
            gc = state.ghost_col[i]
            gr = state.ghost_row[i]
            gf = state.ghost_frightened[i]
            color = jnp.where(gf, _COLOR_FRIGHTENED, _GHOST_COLORS[i])
            frm = draw_tile(frm, gc, gr, color)
            return frm, None

        frame, _ = jax.lax.scan(draw_ghost, frame, jnp.arange(4))

        # --- Fruit ---
        frame = jnp.where(
            state.fruit_active,
            draw_tile(
                frame, jnp.int32(_FRUIT_COL), jnp.int32(_FRUIT_ROW), _COLOR_FRUIT
            ),
            frame,
        )

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Ms. Pac-Man action indices.
            Actions: 0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT.
        """
        import pygame

        return {
            pygame.K_UP: 1,
            pygame.K_w: 1,
            pygame.K_DOWN: 2,
            pygame.K_s: 2,
            pygame.K_LEFT: 3,
            pygame.K_a: 3,
            pygame.K_RIGHT: 4,
            pygame.K_d: 4,
        }
