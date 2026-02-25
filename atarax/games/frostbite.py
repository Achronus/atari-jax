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

"""Frostbite — JAX-native game implementation.

Hop across ice floes to build an igloo while avoiding hazards: fish,
crabs, and falling into freezing water.  Temperature drops constantly;
build the igloo to shelter against the cold.

Action space (18 actions, minimal set):
    0 — NOOP
    1 — FIRE  (jump / toggle direction)
    2 — UP    (jump to floe above)
    3 — RIGHT
    4 — DOWN  (jump to floe below)
    5 — LEFT

Scoring:
    Ice block added to igloo — +10
    Fish caught — +200
    Igloo complete — +1000
    Episode ends when all lives are lost; lives: 4.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_N_ROWS: int = 4  # ice floe rows
_N_FLOES: int = 5  # floes per row
_N_ENEMIES: int = 4  # crabs / fish per row (1 per row)

_ROW_Y = jnp.array([50, 90, 130, 170], dtype=jnp.int32)  # y-centres of floe rows
_IGLOO_X: int = 80
_IGLOO_Y: int = 195
_IGLOO_BLOCKS: int = 8  # blocks needed to complete igloo

_FLOE_SPEED = jnp.array([1.2, -1.0, 0.8, -1.5], dtype=jnp.float32)  # row scroll speeds
_FLOE_W: int = 24  # floe width
_FLOE_SPACING: int = 32  # spacing between floes

_PLAYER_SPEED: float = 2.0
_INIT_LIVES: int = 4

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 60, 100], dtype=jnp.uint8)  # water
_COLOR_FLOE = jnp.array([200, 230, 255], dtype=jnp.uint8)
_COLOR_FLOE_ACTIVE = jnp.array([80, 160, 220], dtype=jnp.uint8)  # floe just jumped
_COLOR_PLAYER = jnp.array([220, 100, 40], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 80, 40], dtype=jnp.uint8)
_COLOR_IGLOO = jnp.array([180, 200, 220], dtype=jnp.uint8)
_COLOR_GROUND = jnp.array([100, 60, 20], dtype=jnp.uint8)


@chex.dataclass
class FrostbiteState(AtariState):
    """
    Complete Frostbite game state — a JAX pytree.

    Parameters
    ----------
    player_row : jax.Array
        int32 — Current floe row (0=top, 3=bottom) or 4=ground.
    player_x : jax.Array
        float32 — Player x on current floe.
    floe_offsets : jax.Array
        float32[4] — Horizontal offset of each row of floes.
    floe_visited : jax.Array
        bool[4, 5] — Which floes have been jumped this cycle.
    igloo_blocks : jax.Array
        int32 — Blocks collected toward igloo completion.
    enemy_x : jax.Array
        float32[4] — Enemy x per row (crab/fish).
    enemy_dir : jax.Array
        int32[4] — Enemy directions.
    temperature : jax.Array
        int32 — Remaining temperature (counts down; die at 0).
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_row: jax.Array
    player_x: jax.Array
    floe_offsets: jax.Array
    floe_visited: jax.Array
    igloo_blocks: jax.Array
    enemy_x: jax.Array
    enemy_dir: jax.Array
    temperature: jax.Array
    key: jax.Array


class Frostbite(AtariEnv):
    """
    Frostbite implemented as a pure JAX function suite.

    Build the igloo before temperature drops to zero.  Lives: 4.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> FrostbiteState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : FrostbiteState
            Player on ground, 4 lives, full temperature.
        """
        return FrostbiteState(
            player_row=jnp.int32(4),  # on ground
            player_x=jnp.float32(80.0),
            floe_offsets=jnp.zeros(_N_ROWS, dtype=jnp.float32),
            floe_visited=jnp.zeros((_N_ROWS, _N_FLOES), dtype=jnp.bool_),
            igloo_blocks=jnp.int32(0),
            enemy_x=jnp.array([20.0, 40.0, 60.0, 80.0], dtype=jnp.float32),
            enemy_dir=jnp.array([1, -1, 1, -1], dtype=jnp.int32),
            temperature=jnp.int32(3000),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: FrostbiteState, action: jax.Array) -> FrostbiteState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : FrostbiteState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : FrostbiteState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Temperature drops
        new_temp = state.temperature - jnp.int32(1)

        # Scroll floes
        new_offsets = state.floe_offsets + _FLOE_SPEED
        new_offsets = new_offsets % jnp.float32(_FLOE_SPACING * _N_FLOES)

        # Player movement
        move_r = action == jnp.int32(3)
        move_l = action == jnp.int32(5)
        new_px = jnp.clip(
            state.player_x
            + jnp.where(move_r, _PLAYER_SPEED, 0.0)
            + jnp.where(move_l, -_PLAYER_SPEED, 0.0),
            5.0,
            155.0,
        )

        # Jump between rows
        jump_up = action == jnp.int32(2)
        jump_dn = action == jnp.int32(4)
        new_row = jnp.clip(
            state.player_row
            - jnp.where(jump_up, jnp.int32(1), jnp.int32(0))
            + jnp.where(jump_dn, jnp.int32(1), jnp.int32(0)),
            0,
            4,
        )

        # Check if player lands on a floe (simplified: always on valid floe)
        on_water = (new_row < 4) & (new_row >= 0)

        # Mark floe as visited; collect block
        floe_idx = (new_px.astype(jnp.int32) // _FLOE_SPACING) % _N_FLOES
        safe_row = jnp.clip(new_row, 0, _N_ROWS - 1)
        was_visited = state.floe_visited[safe_row, floe_idx]
        new_visited = state.floe_visited.at[safe_row, floe_idx].set(True)
        newly_visited = on_water & ~was_visited
        step_reward = step_reward + jnp.where(
            newly_visited, jnp.float32(10.0), jnp.float32(0.0)
        )
        new_igloo = state.igloo_blocks + jnp.where(
            newly_visited, jnp.int32(1), jnp.int32(0)
        )

        # Returning to ground (row 4) with blocks
        at_ground = new_row == jnp.int32(4)
        at_igloo = at_ground & (jnp.abs(new_px - _IGLOO_X) < 20.0)
        igloo_complete = new_igloo >= jnp.int32(_IGLOO_BLOCKS)
        step_reward = step_reward + jnp.where(
            at_igloo & igloo_complete, jnp.float32(1000.0), jnp.float32(0.0)
        )
        new_igloo = jnp.where(at_igloo & igloo_complete, jnp.int32(0), new_igloo)
        new_visited = jnp.where(
            at_igloo, jnp.zeros((_N_ROWS, _N_FLOES), dtype=jnp.bool_), new_visited
        )
        new_temp = jnp.where(at_igloo & igloo_complete, jnp.int32(3000), new_temp)

        # Enemy movement (per row)
        new_ex = state.enemy_x + state.enemy_dir.astype(jnp.float32) * 1.5
        at_edge = (new_ex < 5.0) | (new_ex > 155.0)
        new_edir = jnp.where(at_edge, -state.enemy_dir, state.enemy_dir)
        new_ex = jnp.clip(new_ex, 5.0, 155.0)

        # Enemy collides with player (on same row, same x)
        enemy_rows = jnp.arange(_N_ENEMIES, dtype=jnp.int32)
        enemy_hits = (enemy_rows == new_row) & (jnp.abs(new_ex - new_px) < 12.0)
        hit_enemy = jnp.any(enemy_hits)

        # Fall in water (not on floe = simplified: random chance when jumping)
        fell_in_water = on_water & (
            new_row != state.player_row
        )  # simplified: every jump is safe

        # Life loss
        temp_freeze = new_temp <= jnp.int32(0)
        life_lost = hit_enemy | temp_freeze
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        new_temp = jnp.where(temp_freeze, jnp.int32(3000), new_temp)

        done = new_lives <= jnp.int32(0)

        return FrostbiteState(
            player_row=new_row,
            player_x=new_px,
            floe_offsets=new_offsets,
            floe_visited=new_visited,
            igloo_blocks=new_igloo,
            enemy_x=new_ex,
            enemy_dir=new_edir,
            temperature=new_temp,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: FrostbiteState, action: jax.Array) -> FrostbiteState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : FrostbiteState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : FrostbiteState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: FrostbiteState) -> FrostbiteState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: FrostbiteState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : FrostbiteState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Ground strip
        ground_mask = _ROW_IDX >= 185
        frame = jnp.where(ground_mask[:, :, None], _COLOR_GROUND, frame)

        # Floe rows
        def draw_row_floes(frm, row_i):
            ry = _ROW_Y[row_i]
            offset = state.floe_offsets[row_i].astype(jnp.int32)

            # Draw 5 floes spaced evenly
            def draw_floe(f, fi):
                fx = (offset + fi * _FLOE_SPACING) % (_FLOE_SPACING * _N_FLOES)
                mask = (
                    (_ROW_IDX >= ry - 4)
                    & (_ROW_IDX <= ry + 4)
                    & (_COL_IDX >= fx)
                    & (_COL_IDX < fx + _FLOE_W)
                )
                return jnp.where(mask[:, :, None], _COLOR_FLOE, f), None

            frm, _ = jax.lax.scan(draw_floe, frm, jnp.arange(_N_FLOES))
            return frm, None

        frame, _ = jax.lax.scan(draw_row_floes, frame, jnp.arange(_N_ROWS))

        # Enemies
        def draw_enemy(frm, i):
            ex = state.enemy_x[i].astype(jnp.int32)
            ey = _ROW_Y[i]
            mask = (
                (_ROW_IDX >= ey - 5)
                & (_ROW_IDX <= ey + 5)
                & (_COL_IDX >= ex - 5)
                & (_COL_IDX <= ex + 5)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Igloo (partial construction)
        ig_h = jnp.minimum(state.igloo_blocks, jnp.int32(_IGLOO_BLOCKS)) * 2
        igloo_mask = (
            (_ROW_IDX >= _IGLOO_Y - ig_h)
            & (_ROW_IDX <= _IGLOO_Y)
            & (_COL_IDX >= _IGLOO_X - 12)
            & (_COL_IDX <= _IGLOO_X + 12)
        )
        frame = jnp.where(igloo_mask[:, :, None], _COLOR_IGLOO, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        player_y = jnp.where(
            state.player_row < 4,
            _ROW_Y[jnp.clip(state.player_row, 0, _N_ROWS - 1)] - 8,
            _IGLOO_Y - 10,
        )
        player_mask = (
            (_ROW_IDX >= player_y - 6)
            & (_ROW_IDX <= player_y + 6)
            & (_COL_IDX >= px - 5)
            & (_COL_IDX <= px + 5)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Frostbite action indices.
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
