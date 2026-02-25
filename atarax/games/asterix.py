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

"""Asterix — JAX-native game implementation.

Guide Asterix through a side-scrolling world, collecting magic potions
while avoiding enemies and their cauldrons.  Items scroll from right to
left across 8 fixed lanes.

Action space (5 actions):
    0 — NOOP
    1 — UP    (move lane up)
    2 — DOWN  (move lane down)
    3 — LEFT  (unused; kept for ALE compatibility)
    4 — RIGHT (unused; kept for ALE compatibility)

Scoring:
    Magic potion collected — +50 (doubles each wave)
    Cauldron/enemy hit     — life lost
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
_N_LANES: int = 8  # vertical lanes
_N_ITEMS: int = 8  # concurrent items

# Lane y-positions (pixel centre of each lane)
_LANE_Y = jnp.array(
    [30 + i * 22 for i in range(_N_LANES)], dtype=jnp.int32
)  # [8] → 30, 52, 74, 96, 118, 140, 162, 184

_PLAYER_START_LANE: int = 3  # middle lane
_PLAYER_X: int = 40  # player x (fixed horizontal position)
_ITEM_SCROLL_SPEED: float = 2.0  # px per emulated frame
_ITEM_SPAWN_X: int = 155  # items appear here
_ITEM_SIZE: int = 8  # pixels wide/tall
_PLAYER_SIZE: int = 10  # pixels wide/tall
_DESPAWN_X: int = 0  # remove item when x < this

_SPAWN_INTERVAL_INIT: int = 24  # emulated frames between spawns (decreases with waves)
_WAVE_SPAWN_DECREMENT: int = 2
_ITEMS_PER_WAVE: int = 16  # items collected to advance wave
_INIT_LIVES: int = 3

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_LANE = jnp.array([40, 40, 80], dtype=jnp.uint8)
_COLOR_POTION = jnp.array([100, 220, 100], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([220, 60, 60], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 220, 80], dtype=jnp.uint8)


@chex.dataclass
class AsterixState(AtariState):
    """
    Complete Asterix game state — a JAX pytree.

    Parameters
    ----------
    player_lane : jax.Array
        int32 — Current lane index (0–7).
    item_x : jax.Array
        float32[8] — Item horizontal positions.
    item_lane : jax.Array
        int32[8] — Item lane indices.
    item_is_potion : jax.Array
        bool[8] — True = magic potion (collect); False = enemy (avoid).
    item_active : jax.Array
        bool[8] — Whether the item is currently on screen.
    spawn_timer : jax.Array
        int32 — Sub-steps until next item spawn.
    wave : jax.Array
        int32 — Current wave (increases item speed and score).
    items_collected : jax.Array
        int32 — Potions collected in this wave.
    key : jax.Array
        uint32[2] — PRNG for item type and lane assignment.
    """

    player_lane: jax.Array
    item_x: jax.Array
    item_lane: jax.Array
    item_is_potion: jax.Array
    item_active: jax.Array
    spawn_timer: jax.Array
    wave: jax.Array
    items_collected: jax.Array
    key: jax.Array


class Asterix(AtariEnv):
    """
    Asterix implemented as a pure JAX function suite.

    Collect magic potions; avoid enemies.  Lives: 3.
    """

    num_actions: int = 5

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> AsterixState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : AsterixState
            Player at centre lane, no items active, 3 lives.
        """
        return AsterixState(
            player_lane=jnp.int32(_PLAYER_START_LANE),
            item_x=jnp.full(_N_ITEMS, -20.0, dtype=jnp.float32),
            item_lane=jnp.zeros(_N_ITEMS, dtype=jnp.int32),
            item_is_potion=jnp.zeros(_N_ITEMS, dtype=jnp.bool_),
            item_active=jnp.zeros(_N_ITEMS, dtype=jnp.bool_),
            spawn_timer=jnp.int32(_SPAWN_INTERVAL_INIT),
            wave=jnp.int32(0),
            items_collected=jnp.int32(0),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: AsterixState, action: jax.Array) -> AsterixState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : AsterixState
            Current game state.
        action : jax.Array
            int32 — Action index (0–4).

        Returns
        -------
        new_state : AsterixState
            State after one emulated frame.
        """
        key, sk1, sk2 = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Player lane movement (clamp to valid range)
        new_lane = jnp.clip(
            state.player_lane
            + jnp.where(action == jnp.int32(1), jnp.int32(-1), jnp.int32(0))
            + jnp.where(action == jnp.int32(2), jnp.int32(1), jnp.int32(0)),
            0,
            _N_LANES - 1,
        )

        # Scroll items left
        speed = _ITEM_SCROLL_SPEED + state.wave.astype(jnp.float32) * 0.5
        new_item_x = state.item_x - speed
        # Deactivate items that have scrolled off screen
        new_active = state.item_active & (new_item_x >= jnp.float32(_DESPAWN_X))

        # Spawn new item
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = new_spawn_timer <= jnp.int32(0)
        spawn_interval = jnp.maximum(
            jnp.int32(6),
            jnp.int32(_SPAWN_INTERVAL_INIT)
            - state.wave * jnp.int32(_WAVE_SPAWN_DECREMENT),
        )
        new_spawn_timer = jnp.where(do_spawn, spawn_interval, new_spawn_timer)

        # Find first inactive slot for spawn
        slot = jnp.argmin(new_active.astype(jnp.int32))  # first False → 0
        new_item_lane = jax.random.randint(sk1, (), 0, _N_LANES)
        # ~50% potions, increasing with wave
        potion_prob = jnp.minimum(0.75, 0.4 + state.wave.astype(jnp.float32) * 0.05)
        is_potion = jax.random.uniform(sk2) < potion_prob

        new_item_x = jnp.where(
            do_spawn,
            new_item_x.at[slot].set(jnp.float32(_ITEM_SPAWN_X)),
            new_item_x,
        )
        new_item_lane_arr = jnp.where(
            do_spawn,
            state.item_lane.at[slot].set(new_item_lane),
            state.item_lane,
        )
        new_item_is_potion = jnp.where(
            do_spawn,
            state.item_is_potion.at[slot].set(is_potion),
            state.item_is_potion,
        )
        new_active = jnp.where(
            do_spawn,
            new_active.at[slot].set(True),
            new_active,
        )

        # Collision: item at player x and lane
        player_y = _LANE_Y[new_lane]
        item_y = _LANE_Y[new_item_lane_arr]  # [8]
        near_x = (new_item_x >= jnp.float32(_PLAYER_X - _ITEM_SIZE)) & (
            new_item_x <= jnp.float32(_PLAYER_X + _PLAYER_SIZE)
        )
        near_y = item_y == player_y
        hit = new_active & near_x & near_y

        # Collect potions
        collected = hit & new_item_is_potion
        n_collected = jnp.sum(collected).astype(jnp.int32)
        potion_score = jnp.float32(50) * jnp.float32(2**state.wave)
        step_reward = step_reward + jnp.float32(n_collected) * potion_score

        # Hit enemies
        hit_enemy = hit & ~new_item_is_potion
        any_hit_enemy = jnp.any(hit_enemy)
        new_lives = state.lives - jnp.where(any_hit_enemy, jnp.int32(1), jnp.int32(0))

        # Remove collected/hit items
        new_active = new_active & ~hit

        # Wave progression
        new_items_collected = state.items_collected + n_collected
        wave_up = new_items_collected >= jnp.int32(_ITEMS_PER_WAVE)
        new_wave = state.wave + jnp.where(wave_up, jnp.int32(1), jnp.int32(0))
        new_items_collected = jnp.where(wave_up, jnp.int32(0), new_items_collected)

        done = new_lives <= jnp.int32(0)

        return AsterixState(
            player_lane=new_lane,
            item_x=new_item_x,
            item_lane=new_item_lane_arr,
            item_is_potion=new_item_is_potion,
            item_active=new_active,
            spawn_timer=new_spawn_timer,
            wave=new_wave,
            items_collected=new_items_collected,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: AsterixState, action: jax.Array) -> AsterixState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : AsterixState
            Current game state.
        action : jax.Array
            int32 — Action index (0–4).

        Returns
        -------
        new_state : AsterixState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: AsterixState) -> AsterixState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: AsterixState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : AsterixState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Draw lane dividers
        def draw_lane_line(frm, i):
            y = _LANE_Y[i]
            mask = (_ROW_IDX == y) & (_COL_IDX < 160)
            return jnp.where(mask[:, :, None], _COLOR_LANE, frm), None

        frame, _ = jax.lax.scan(draw_lane_line, frame, jnp.arange(_N_LANES))

        # Draw items
        def draw_item(frm, i):
            ix = state.item_x[i].astype(jnp.int32)
            iy = _LANE_Y[state.item_lane[i]]
            color = jnp.where(state.item_is_potion[i], _COLOR_POTION, _COLOR_ENEMY)
            mask = (
                state.item_active[i]
                & (_ROW_IDX >= iy - _ITEM_SIZE // 2)
                & (_ROW_IDX <= iy + _ITEM_SIZE // 2)
                & (_COL_IDX >= ix)
                & (_COL_IDX < ix + _ITEM_SIZE)
            )
            return jnp.where(mask[:, :, None], color, frm), None

        frame, _ = jax.lax.scan(draw_item, frame, jnp.arange(_N_ITEMS))

        # Draw player
        py = _LANE_Y[state.player_lane]
        player_mask = (
            (_ROW_IDX >= py - _PLAYER_SIZE // 2)
            & (_ROW_IDX <= py + _PLAYER_SIZE // 2)
            & (_COL_IDX >= _PLAYER_X)
            & (_COL_IDX < _PLAYER_X + _PLAYER_SIZE)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Asterix action indices.
        """
        import pygame

        return {
            pygame.K_UP: 1,
            pygame.K_w: 1,
            pygame.K_DOWN: 2,
            pygame.K_s: 2,
        }
