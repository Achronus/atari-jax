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

"""Ms. Pac-Man — JAX-native SDF game implementation.

Characters move one tile per agent step; no 4-frame skip.
Maze: 31 rows × 28 cols tile grid, each tile = 5×5 pixels.
      Centred in the 160×210 frame (x-offset=10, y-offset=27).

Action space (9 actions, ALE minimal set):
    0 — NOOP (continue current direction)
    1 — UP
    2 — RIGHT
    3 — LEFT
    4 — DOWN
    5 — UPRIGHT (treated as UP)
    6 — UPLEFT  (treated as UP)
    7 — DOWNRIGHT (treated as DOWN)
    8 — DOWNLEFT  (treated as DOWN)
"""

from typing import ClassVar

import chex
import jax.numpy as jnp
import numpy as np

from atarax.env._base.maze_navigator import MazeNavigatorGame, MazeNavigatorState
from atarax.env.sdf import (
    finalise_rgb,
    make_canvas,
    paint_layer,
    paint_sdf,
    render_bool_grid,
    sdf_circle,
    sdf_ghost,
    sdf_subtract,
    sdf_triangle,
)
from atarax.game import AtaraxParams

# ── Maze dimensions
_ROWS: int = 31
_COLS: int = 28
_TILE_W: float = 5.0
_TILE_H: float = 5.0
_OFFSET_X: float = 10.0
_OFFSET_Y: float = 27.0

# ── Start positions
_PAC_ROW: int = 23
_PAC_COL: int = 13

# Ghost start positions: Blinky(red), Pinky(pink), Inky(cyan), Sue(orange)
_GHOST_ROWS: tuple = (11, 11, 13, 13)
_GHOST_COLS: tuple = (13, 14, 13, 14)

# ── Power pellet tile positions (2 in Ms. Pac-Man)
_POWER_ROWS: tuple = (23, 23)
_POWER_COLS: tuple = (4, 23)

# ── Fruit
_FRUIT_ROW: int = 14
_FRUIT_COL: int = 13
_FRUIT_SCORE: int = 100

# ── Ghost scatter corners in frightened mode: Blinky, Pinky, Inky, Sue
_FRIGHT_TR = jnp.array([0, 0, _ROWS - 1, _ROWS - 1], dtype=jnp.int32)
_FRIGHT_TC = jnp.array([0, _COLS - 1, 0, _COLS - 1], dtype=jnp.int32)

# ── Direction deltas (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT) — mirrors maze_navigator.py
_DR = jnp.array([-1, 0, 1, 0], dtype=jnp.int32)
_DC = jnp.array([0, 1, 0, -1], dtype=jnp.int32)

# ── Action → direction
# NOOP(0)→keep; UP(1)→0, RIGHT(2)→1, LEFT(3)→3, DOWN(4)→2
# UPRIGHT(5)→0, UPLEFT(6)→0, DOWNRIGHT(7)→2, DOWNLEFT(8)→2
_ACTION_TO_DIR = jnp.array([0, 0, 1, 3, 2, 0, 0, 2, 2], dtype=jnp.int32)

# ── Ghost score chain: indexed by combo_count - 1 (capped at 3)
_GHOST_SCORES = jnp.array([200, 400, 800, 1600], dtype=jnp.int32)

# ── Facing angle per direction (radians): UP, RIGHT, DOWN, LEFT
_DIR_ANGLE = jnp.array([-jnp.pi / 2, 0.0, jnp.pi / 2, jnp.pi], dtype=jnp.float32)


# ── Maze layout (0=open, 1=wall)
def _build_tile_map() -> np.ndarray:
    rows = [
        "############################",  # 0
        "#............##............#",  # 1
        "#.####.#####.##.#####.####.#",  # 2
        "#.####.#####.##.#####.####.#",  # 3
        "#.####.#####.##.#####.####.#",  # 4
        "#..........................#",  # 5
        "#.####.##.########.##.####.#",  # 6
        "#.####.##.########.##.####.#",  # 7
        "#......##....##....##......#",  # 8
        "######.#####.##.#####.######",  # 9
        "######.#####.##.#####.######",  # 10
        "######.##..........##.######",  # 11  Blinky (col 13), Pinky (col 14)
        "######.##.###..###.##.######",  # 12  ghost house entrance (cols 13-14 open)
        "######.##.#......#.##.######",  # 13  ghost house interior: Inky (col 13), Sue (col 14)
        "######....#......#....######",  # 14  ghost house exit corridor
        "######.##.########.##.######",  # 15
        "######.##..........##.######",  # 16
        "######.##.########.##.######",  # 17
        "######.#####.##.#####.######",  # 18
        "#............##............#",  # 19
        "#.####.#####.##.#####.####.#",  # 20
        "#.####.#####.##.#####.####.#",  # 21
        "#.####.#####.##.#####.####.#",  # 22
        "#..........................#",  # 23  Pac-Man (col 13), power pellets col 4 & 23
        "###.##.##.########.##.##.###",  # 24
        "###.##.##.########.##.##.###",  # 25
        "#......##....##....##......#",  # 26
        "#.##########.##.##########.#",  # 27
        "#.##########.##.##########.#",  # 28
        "#..........................#",  # 29
        "############################",  # 30
    ]
    m = np.zeros((_ROWS, _COLS), dtype=np.int32)
    for r, s in enumerate(rows):
        for c, ch in enumerate(s):
            m[r, c] = 1 if ch == "#" else 0
    return m


_TILE_MAP_NP = _build_tile_map()
_TILE_MAP = jnp.array(_TILE_MAP_NP, dtype=jnp.int32)


def _build_init_collectibles() -> np.ndarray:
    coll = (1 - _TILE_MAP_NP).astype(bool)
    coll[11:16, :] = False  # ghost house zone
    coll[_PAC_ROW, _PAC_COL] = False  # Ms. Pac-Man start tile
    return coll


_INIT_COLLECTIBLES_NP = _build_init_collectibles()
_INIT_COLLECTIBLES = jnp.array(_INIT_COLLECTIBLES_NP, dtype=jnp.bool_)

# ── Colours (float32 RGB in [0, 1])
_COL_BG = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
_COL_WALL = jnp.array([0.078, 0.078, 0.784], dtype=jnp.float32)
_COL_DOT = jnp.array([1.0, 0.902, 0.725], dtype=jnp.float32)
_COL_POWER = jnp.array([1.0, 1.0, 0.784], dtype=jnp.float32)
_COL_MSPACMAN = jnp.array([1.0, 0.90, 0.0], dtype=jnp.float32)
_COL_BLINKY = jnp.array([0.835, 0.196, 0.196], dtype=jnp.float32)
_COL_PINKY = jnp.array([1.0, 0.718, 0.80], dtype=jnp.float32)
_COL_INKY = jnp.array([0.0, 0.863, 0.784], dtype=jnp.float32)
_COL_SUE = jnp.array([1.0, 0.502, 0.0], dtype=jnp.float32)
_COL_GHOST_FRIGHT = jnp.array([0.0, 0.0, 0.784], dtype=jnp.float32)
_COL_FRUIT = jnp.array([1.0, 0.314, 0.471], dtype=jnp.float32)

_GHOST_COLOURS = [_COL_BLINKY, _COL_PINKY, _COL_INKY, _COL_SUE]


@chex.dataclass
class MsPacManParams(AtaraxParams):
    """
    Static configuration for Ms. Pac-Man.

    Parameters
    ----------
    max_steps : int
        Maximum agent steps per episode.
    fright_duration : int
        Steps a power pellet keeps ghosts frightened.
    fruit_trigger : int
        Dots eaten before fruit spawns.
    fruit_duration : int
        Steps the fruit remains before despawning.
    """

    max_steps: int = 18000
    fright_duration: int = 30
    fruit_trigger: int = 70
    fruit_duration: int = 20


@chex.dataclass
class MsPacManState(MazeNavigatorState):
    """
    Ms. Pac-Man game state.

    Extends `MazeNavigatorState` with ghost-eat combo, fruit tracking,
    and dot counter.

    Inherited from `MazeNavigatorState`:
        `player_row`, `player_col`, `player_dir`, `tile_map`,
        `collectibles`, `power_timer`, `ghost_row`, `ghost_col`,
        `ghost_dir`, `ghost_mode`.

    Inherited from `AtariState`:
        `reward`, `done`, `step`, `episode_step`, `lives`, `score`, `level`, `key`.

    Parameters
    ----------
    combo_count : chex.Array
        int32 — ghost-eat chain counter; resets to 1 on each power pellet.
    fruit_active : chex.Array
        bool — `True` while fruit is on the board.
    fruit_timer : chex.Array
        int32 — steps until the fruit despawns.
    dots_eaten : chex.Array
        int32 — total collectibles eaten this episode (triggers fruit spawn).
    """

    combo_count: chex.Array
    fruit_active: chex.Array
    fruit_timer: chex.Array
    dots_eaten: chex.Array


class MsPacMan(MazeNavigatorGame):
    """
    Ms. Pac-Man implemented as a pure-JAX function suite.

    All four ghosts (Blinky, Pinky, Inky, Sue) chase Ms. Pac-Man by
    minimising L1 distance. When frightened they scatter to corners.
    Characters move one tile per agent step. No 4-frame skip.
    """

    num_actions: int = 9
    game_id: ClassVar[str] = "ms_pacman"

    def _reset(self, rng: chex.PRNGKey) -> MsPacManState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.

        Returns
        -------
        state : MsPacManState
            Full pellet grid, Ms. Pac-Man and ghosts at start positions, 3 lives.
        """
        ghost_dirs = jnp.array([2, 2, 2, 2], dtype=jnp.int32)  # all face DOWN initially
        ghost_mode = jnp.zeros(4, dtype=jnp.int32)
        return MsPacManState(
            # MazeNavigatorState fields
            player_row=jnp.int32(_PAC_ROW),
            player_col=jnp.int32(_PAC_COL),
            player_dir=jnp.int32(3),  # start facing LEFT
            tile_map=_TILE_MAP,
            collectibles=_INIT_COLLECTIBLES,
            power_timer=jnp.int32(0),
            ghost_row=jnp.array(_GHOST_ROWS, dtype=jnp.int32),
            ghost_col=jnp.array(_GHOST_COLS, dtype=jnp.int32),
            ghost_dir=ghost_dirs,
            ghost_mode=ghost_mode,
            # MsPacManState fields
            combo_count=jnp.int32(1),
            fruit_active=jnp.bool_(False),
            fruit_timer=jnp.int32(0),
            dots_eaten=jnp.int32(0),
            # AtariState fields
            lives=jnp.int32(3),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=rng,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: MsPacManState,
        action: chex.Array,
        params: MsPacManParams,
    ) -> MsPacManState:
        """
        Advance the game by one agent step (one tile movement per character).

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key (unused; kept for interface compatibility).
        state : MsPacManState
            Current game state.
        action : chex.Array
            int32 — Action index (0=NOOP, 1=UP, 2=RIGHT, 3=LEFT, 4=DOWN,
            5=UPRIGHT, 6=UPLEFT, 7=DOWNRIGHT, 8=DOWNLEFT).
        params : MsPacManParams
            Static environment parameters.

        Returns
        -------
        new_state : MsPacManState
            State after one tile step.
        """
        tile_map = state.tile_map

        # ── 1. Player movement
        requested_dir = jnp.where(
            action == jnp.int32(0),
            state.player_dir,
            _ACTION_TO_DIR[action],
        )
        can_move_req = self._can_move(
            state.player_row, state.player_col, requested_dir, tile_map
        )
        new_dir = jnp.where(can_move_req, requested_dir, state.player_dir)
        new_pac_row, new_pac_col = self._step_entity(
            state.player_row, state.player_col, new_dir, tile_map
        )

        # ── 2. Collectible pickup
        has_coll = state.collectibles[new_pac_row, new_pac_col]
        is_power = (
            (new_pac_row == jnp.int32(_POWER_ROWS[0]))
            & (new_pac_col == jnp.int32(_POWER_COLS[0]))
        ) | (
            (new_pac_row == jnp.int32(_POWER_ROWS[1]))
            & (new_pac_col == jnp.int32(_POWER_COLS[1]))
        )
        ate_dot = has_coll & ~is_power
        ate_power = has_coll & is_power

        new_collectibles = jnp.where(
            has_coll,
            state.collectibles.at[new_pac_row, new_pac_col].set(jnp.bool_(False)),
            state.collectibles,
        )

        dot_reward = jnp.where(ate_dot, jnp.int32(10), jnp.int32(0))
        power_reward = jnp.where(ate_power, jnp.int32(50), jnp.int32(0))

        new_power_timer = jnp.where(
            ate_power,
            jnp.int32(params.fright_duration),
            jnp.maximum(state.power_timer - jnp.int32(1), jnp.int32(0)),
        )
        new_combo = jnp.where(ate_power, jnp.int32(1), state.combo_count)
        new_dots_eaten = state.dots_eaten + jnp.where(
            ate_dot | ate_power, jnp.int32(1), jnp.int32(0)
        )

        # ── 3. Ghost movement — all 4 ghosts chase directly (L1 to Ms. Pac-Man)
        frightened = new_power_timer > jnp.int32(0)

        ghost_rows = state.ghost_row
        ghost_cols = state.ghost_col
        ghost_dirs = state.ghost_dir

        for g in range(4):
            target_r = jnp.where(frightened, _FRIGHT_TR[g], new_pac_row)
            target_c = jnp.where(frightened, _FRIGHT_TC[g], new_pac_col)
            new_dir_g = self._pick_ghost_direction(
                ghost_rows[g],
                ghost_cols[g],
                ghost_dirs[g],
                target_r,
                target_c,
                tile_map,
            )
            new_r, new_c = self._step_entity(
                ghost_rows[g], ghost_cols[g], new_dir_g, tile_map
            )
            ghost_rows = ghost_rows.at[g].set(new_r)
            ghost_cols = ghost_cols.at[g].set(new_c)
            ghost_dirs = ghost_dirs.at[g].set(new_dir_g)

        # ── 4. Ghost–player collisions
        ghost_reward = jnp.int32(0)
        life_lost = jnp.bool_(False)

        for g in range(4):
            hit = (ghost_rows[g] == new_pac_row) & (ghost_cols[g] == new_pac_col)
            can_eat = frightened & hit
            can_die = (~frightened) & hit

            ghost_score = _GHOST_SCORES[
                jnp.clip(new_combo - jnp.int32(1), jnp.int32(0), jnp.int32(3))
            ]
            ghost_reward = ghost_reward + jnp.where(can_eat, ghost_score, jnp.int32(0))
            new_combo = new_combo + jnp.where(can_eat, jnp.int32(1), jnp.int32(0))
            life_lost = life_lost | can_die

            ghost_rows = ghost_rows.at[g].set(
                jnp.where(can_eat, jnp.int32(_GHOST_ROWS[g]), ghost_rows[g])
            )
            ghost_cols = ghost_cols.at[g].set(
                jnp.where(can_eat, jnp.int32(_GHOST_COLS[g]), ghost_cols[g])
            )

        # ── 5. Death: reset positions
        _start_rows = jnp.array(_GHOST_ROWS, dtype=jnp.int32)
        _start_cols = jnp.array(_GHOST_COLS, dtype=jnp.int32)

        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        new_pac_row = jnp.where(life_lost, jnp.int32(_PAC_ROW), new_pac_row)
        new_pac_col = jnp.where(life_lost, jnp.int32(_PAC_COL), new_pac_col)
        ghost_rows = jnp.where(life_lost, _start_rows, ghost_rows)
        ghost_cols = jnp.where(life_lost, _start_cols, ghost_cols)
        new_power_timer = jnp.where(life_lost, jnp.int32(0), new_power_timer)

        # ── 6. Level clear
        all_cleared = ~jnp.any(new_collectibles)
        new_level = state.level + jnp.where(all_cleared, jnp.int32(1), jnp.int32(0))
        new_collectibles = jnp.where(all_cleared, _INIT_COLLECTIBLES, new_collectibles)
        new_pac_row = jnp.where(all_cleared, jnp.int32(_PAC_ROW), new_pac_row)
        new_pac_col = jnp.where(all_cleared, jnp.int32(_PAC_COL), new_pac_col)
        ghost_rows = jnp.where(all_cleared, _start_rows, ghost_rows)
        ghost_cols = jnp.where(all_cleared, _start_cols, ghost_cols)

        # ── 7. Fruit spawn / collect / despawn
        spawn_fruit = (
            new_dots_eaten >= jnp.int32(params.fruit_trigger)
        ) & ~state.fruit_active
        new_fruit_timer = jnp.where(
            spawn_fruit,
            jnp.int32(params.fruit_duration),
            jnp.where(
                state.fruit_active,
                state.fruit_timer - jnp.int32(1),
                state.fruit_timer,
            ),
        )
        new_fruit_active = (state.fruit_active | spawn_fruit) & (
            new_fruit_timer > jnp.int32(0)
        )

        fruit_eaten = (
            new_fruit_active
            & (new_pac_row == jnp.int32(_FRUIT_ROW))
            & (new_pac_col == jnp.int32(_FRUIT_COL))
        )
        fruit_reward = jnp.where(fruit_eaten, jnp.int32(_FRUIT_SCORE), jnp.int32(0))
        new_fruit_active = new_fruit_active & ~fruit_eaten

        # ── 8. Terminal and reward
        done = (new_lives <= jnp.int32(0)) | (
            state.episode_step + jnp.int32(1) >= jnp.int32(params.max_steps)
        )
        total_reward = (dot_reward + power_reward + ghost_reward + fruit_reward).astype(
            jnp.float32
        )
        new_score = (
            state.score + dot_reward + power_reward + ghost_reward + fruit_reward
        )

        ghost_mode = jnp.where(
            new_power_timer > jnp.int32(0),
            jnp.full(4, jnp.int32(2), dtype=jnp.int32),
            jnp.zeros(4, dtype=jnp.int32),
        )

        return state.__replace__(
            player_row=new_pac_row,
            player_col=new_pac_col,
            player_dir=new_dir,
            collectibles=new_collectibles,
            power_timer=new_power_timer,
            ghost_row=ghost_rows,
            ghost_col=ghost_cols,
            ghost_dir=ghost_dirs,
            ghost_mode=ghost_mode,
            combo_count=new_combo,
            fruit_active=new_fruit_active,
            fruit_timer=new_fruit_timer,
            dots_eaten=new_dots_eaten,
            lives=new_lives,
            score=new_score,
            level=new_level,
            reward=total_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def render(self, state: MsPacManState) -> chex.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : MsPacManState
            Current game state.

        Returns
        -------
        frame : chex.Array
            uint8[210, 160, 3] — RGB image.
        """
        canvas = make_canvas(_COL_BG)

        # Layer 1 — Walls
        wall_mask = render_bool_grid(
            state.tile_map.astype(jnp.bool_),
            cell_x0=_OFFSET_X,
            cell_y0=_OFFSET_Y,
            cell_w=_TILE_W,
            cell_h=_TILE_H,
        )
        canvas = paint_layer(canvas, wall_mask, _COL_WALL)

        # Layer 2 — Dots (small rects at tile centres)
        dot_mask = render_bool_grid(
            state.collectibles,
            cell_x0=_OFFSET_X,
            cell_y0=_OFFSET_Y,
            cell_w=_TILE_W * 0.3,
            cell_h=_TILE_H * 0.3,
        )
        canvas = paint_layer(canvas, dot_mask, _COL_DOT)

        # Layer 3 — Power pellets (2 larger circles at fixed positions)
        for i in range(2):
            pr, pc = _POWER_ROWS[i], _POWER_COLS[i]
            pp_active = state.collectibles[pr, pc]
            pp_cx = _OFFSET_X + pc * _TILE_W + _TILE_W * 0.5
            pp_cy = _OFFSET_Y + pr * _TILE_H + _TILE_H * 0.5
            pp_mask = (sdf_circle(pp_cx, pp_cy, 3.0) < 0.0) & pp_active
            canvas = paint_layer(canvas, pp_mask, _COL_POWER)

        # Layer 4 — Fruit (JAX-gated by fruit_active)
        fr_cx = _OFFSET_X + _FRUIT_COL * _TILE_W + _TILE_W * 0.5
        fr_cy = _OFFSET_Y + _FRUIT_ROW * _TILE_H + _TILE_H * 0.5
        fr_mask = (sdf_circle(fr_cx, fr_cy, 2.5) < 0.0) & state.fruit_active
        canvas = paint_layer(canvas, fr_mask, _COL_FRUIT)

        # Layer 5 — Ghosts (body; colour switches to blue when frightened)
        frightened = state.power_timer > jnp.int32(0)
        for g in range(4):
            gx, gy = self._grid_to_pixel(
                state.ghost_row[g],
                state.ghost_col[g],
                jnp.float32(_TILE_H),
                jnp.float32(_TILE_W),
                jnp.float32(_OFFSET_Y),
                jnp.float32(_OFFSET_X),
            )
            ghost_colour = jnp.where(frightened, _COL_GHOST_FRIGHT, _GHOST_COLOURS[g])
            canvas = paint_sdf(canvas, sdf_ghost(gx, gy, r=4.0), ghost_colour)

        # Layer 6 — Ms. Pac-Man with animated mouth
        px, py = self._grid_to_pixel(
            state.player_row,
            state.player_col,
            jnp.float32(_TILE_H),
            jnp.float32(_TILE_W),
            jnp.float32(_OFFSET_Y),
            jnp.float32(_OFFSET_X),
        )
        mouth_phase = (state.step % jnp.int32(8)).astype(jnp.float32) / jnp.float32(8.0)
        mouth_angle = jnp.maximum(
            jnp.abs(mouth_phase - jnp.float32(0.5)) * jnp.float32(1.2),
            jnp.float32(0.05),
        )
        pac_facing = _DIR_ANGLE[state.player_dir]
        tip_r = jnp.float32(5.5)
        mouth = sdf_triangle(
            px,
            py,
            px + jnp.cos(pac_facing + mouth_angle) * tip_r,
            py + jnp.sin(pac_facing + mouth_angle) * tip_r,
            px + jnp.cos(pac_facing - mouth_angle) * tip_r,
            py + jnp.sin(pac_facing - mouth_angle) * tip_r,
        )
        pac_body = sdf_subtract(sdf_circle(px, py, jnp.float32(3.5)), mouth)
        canvas = paint_sdf(canvas, pac_body, _COL_MSPACMAN)

        return finalise_rgb(canvas)
