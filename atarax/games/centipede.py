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

"""Centipede — JAX-native game implementation.

Shoot a multi-segment centipede as it zigzags down through a mushroom
field.  Each segment hit leaves a mushroom; the head bounces off mushrooms
and walls.  A spider, flea, and scorpion make occasional appearances.

Action space (6 actions):
    0 — NOOP
    1 — FIRE
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT

Scoring:
    Centipede segment (body) — +10
    Centipede head           — +100
    Mushroom                 — +1 (when shot)
    Spider                   — +300–900
    Flea                     — +200
    Lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Grid layout (16×16 tiles, each 10×10 px)
# ---------------------------------------------------------------------------
_TILE: int = 10
_GRID_COLS: int = 15
_GRID_ROWS: int = 19
_GRID_X0: int = 5
_GRID_Y0: int = 0

_N_SEG: int = 10  # centipede segments
_SEG_W: int = 8
_SEG_H: int = 8

_PLAYER_ZONE_ROWS: int = 5  # player confined to bottom rows
_PLAYER_W: int = 8
_PLAYER_H: int = 6

_BULLET_W: int = 2
_BULLET_H: int = 6
_BULLET_SPEED: float = 6.0

_CENT_SPEED: float = 1.5  # px per sub-step horizontal
_FLEA_SPEED: float = 2.0

_N_MUSH: int = 30  # number of mushrooms

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_MUSH = jnp.array([200, 0, 200], dtype=jnp.uint8)
_COLOR_MUSH_HIT = jnp.array([100, 0, 100], dtype=jnp.uint8)
_COLOR_CENTIPEDE = jnp.array([0, 220, 0], dtype=jnp.uint8)
_COLOR_HEAD = jnp.array([255, 255, 0], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 255, 255], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 200, 0], dtype=jnp.uint8)
_COLOR_SPIDER = jnp.array([255, 100, 0], dtype=jnp.uint8)
_COLOR_FLEA = jnp.array([200, 200, 200], dtype=jnp.uint8)

# Initial mushroom x/y positions (pixel coords)
_MUSH_INIT_X = jnp.array(
    [
        15,
        35,
        55,
        75,
        95,
        115,
        135,
        25,
        45,
        65,
        85,
        105,
        125,
        145,
        20,
        40,
        60,
        80,
        100,
        120,
        140,
        30,
        50,
        70,
        90,
        110,
        130,
        10,
        70,
        130,
    ],
    dtype=jnp.float32,
)
_MUSH_INIT_Y = jnp.array(
    [
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        60,
        60,
        60,
        60,
        60,
        60,
        60,
        80,
        80,
        80,
        80,
        80,
        80,
        100,
        100,
        100,
    ],
    dtype=jnp.float32,
)


@chex.dataclass
class CentipedeState(AtariState):
    """
    Complete Centipede game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x (left edge), confined to bottom zone.
    player_y : jax.Array
        float32 — Player y.
    bullet_x : jax.Array
        float32 — Bullet x.
    bullet_y : jax.Array
        float32 — Bullet y.
    bullet_active : jax.Array
        bool — Bullet in flight.
    seg_x : jax.Array
        float32[10] — Segment x positions.
    seg_y : jax.Array
        float32[10] — Segment y positions.
    seg_alive : jax.Array
        bool[10] — Alive segments.
    seg_dx : jax.Array
        float32[10] — Per-segment horizontal velocity.
    mush_x : jax.Array
        float32[30] — Mushroom x positions.
    mush_y : jax.Array
        float32[30] — Mushroom y positions.
    mush_hp : jax.Array
        int32[30] — Mushroom HP (0=destroyed, 4=full).
    spider_x : jax.Array
        float32 — Spider x.
    spider_y : jax.Array
        float32 — Spider y.
    spider_active : jax.Array
        bool.
    spider_timer : jax.Array
        int32 — Sub-steps until spider spawns.
    flea_x : jax.Array
        float32 — Flea x.
    flea_y : jax.Array
        float32 — Flea y.
    flea_active : jax.Array
        bool.
    flea_timer : jax.Array
        int32.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    seg_x: jax.Array
    seg_y: jax.Array
    seg_alive: jax.Array
    seg_dx: jax.Array
    mush_x: jax.Array
    mush_y: jax.Array
    mush_hp: jax.Array
    spider_x: jax.Array
    spider_y: jax.Array
    spider_active: jax.Array
    spider_timer: jax.Array
    flea_x: jax.Array
    flea_y: jax.Array
    flea_active: jax.Array
    flea_timer: jax.Array
    key: jax.Array


_PLAYER_ZONE_TOP: float = float(_GRID_ROWS - _PLAYER_ZONE_ROWS) * _TILE
_PLAYER_START_X: float = 72.0
_PLAYER_START_Y: float = _PLAYER_ZONE_TOP + 10.0


class Centipede(AtariEnv):
    """
    Centipede implemented as a pure JAX function suite.

    Shoot all centipede segments to advance to the next wave.  Lives: 3.
    """

    num_actions: int = 6

    def _reset(self, key: jax.Array) -> CentipedeState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : CentipedeState
            Centipede at top, player at bottom, mushrooms scattered.
        """
        # Centipede starts in top-right moving left
        seg_x = jnp.array(
            [130.0, 120.0, 110.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0],
            dtype=jnp.float32,
        )
        seg_y = jnp.full(10, 10.0, dtype=jnp.float32)
        seg_dx = jnp.full(10, -_CENT_SPEED, dtype=jnp.float32)
        return CentipedeState(
            player_x=jnp.float32(_PLAYER_START_X),
            player_y=jnp.float32(_PLAYER_START_Y),
            bullet_x=jnp.float32(80.0),
            bullet_y=jnp.float32(_PLAYER_START_Y),
            bullet_active=jnp.bool_(False),
            seg_x=seg_x,
            seg_y=seg_y,
            seg_alive=jnp.ones(10, dtype=jnp.bool_),
            seg_dx=seg_dx,
            mush_x=_MUSH_INIT_X,
            mush_y=_MUSH_INIT_Y,
            mush_hp=jnp.full(_N_MUSH, 4, dtype=jnp.int32),
            spider_x=jnp.float32(0.0),
            spider_y=jnp.float32(_PLAYER_START_Y),
            spider_active=jnp.bool_(False),
            spider_timer=jnp.int32(150),
            flea_x=jnp.float32(80.0),
            flea_y=jnp.float32(0.0),
            flea_active=jnp.bool_(False),
            flea_timer=jnp.int32(200),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: CentipedeState, action: jax.Array) -> CentipedeState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : CentipedeState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : CentipedeState
            State after one emulated frame.
        """
        key, k_sp, k_fl = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Player movement (confined to bottom zone)
        pdx = jnp.where(
            action == jnp.int32(3),
            jnp.float32(2.0),
            jnp.where(action == jnp.int32(5), jnp.float32(-2.0), jnp.float32(0.0)),
        )
        pdy = jnp.where(
            action == jnp.int32(4),
            jnp.float32(2.0),
            jnp.where(action == jnp.int32(2), jnp.float32(-2.0), jnp.float32(0.0)),
        )
        new_px = jnp.clip(state.player_x + pdx, jnp.float32(5.0), jnp.float32(147.0))
        new_py = jnp.clip(
            state.player_y + pdy, jnp.float32(_PLAYER_ZONE_TOP), jnp.float32(195.0)
        )

        # Fire bullet
        fire = (action == jnp.int32(1)) & ~state.bullet_active
        new_bx = jnp.where(fire, new_px + _PLAYER_W / 2, state.bullet_x)
        new_by = jnp.where(fire, new_py, state.bullet_y)
        new_bactive = state.bullet_active | fire
        new_by = jnp.where(new_bactive, new_by - _BULLET_SPEED, new_by)
        new_bactive = new_bactive & (new_by > jnp.float32(0.0))

        # Bullet hits mushrooms
        mush_bx = jnp.abs(state.mush_x - new_bx) < jnp.float32(6.0)
        mush_by = jnp.abs(state.mush_y - new_by) < jnp.float32(6.0)
        mush_hit_by_bullet = (
            mush_bx & mush_by & new_bactive & (state.mush_hp > jnp.int32(0))
        )
        new_mush_hp = jnp.where(
            mush_hit_by_bullet, state.mush_hp - jnp.int32(1), state.mush_hp
        )
        mush_destroyed = mush_hit_by_bullet & (new_mush_hp <= jnp.int32(0))
        step_reward = step_reward + jnp.float32(
            jnp.sum(mush_destroyed).astype(jnp.int32)
        )
        any_mush_hit = jnp.any(mush_hit_by_bullet)
        new_bactive = new_bactive & ~any_mush_hit

        # Centipede segment movement
        new_seg_x = jnp.where(state.seg_alive, state.seg_x + state.seg_dx, state.seg_x)
        new_seg_y = state.seg_y
        new_seg_dx = state.seg_dx

        # Bounce off walls or mushrooms for each segment
        hit_right = new_seg_x + _SEG_W > jnp.float32(150.0)
        hit_left = new_seg_x < jnp.float32(5.0)
        wall_bounce = hit_right | hit_left
        new_seg_x = jnp.where(
            wall_bounce & state.seg_alive, jnp.clip(new_seg_x, 5.0, 142.0), new_seg_x
        )
        new_seg_y = jnp.where(
            wall_bounce & state.seg_alive, new_seg_y + jnp.float32(_TILE), new_seg_y
        )
        new_seg_dx = jnp.where(wall_bounce & state.seg_alive, -new_seg_dx, new_seg_dx)

        # Centipede reaches bottom: reset to top
        at_bottom = new_seg_y > jnp.float32(_PLAYER_ZONE_TOP)
        new_seg_y = jnp.where(state.seg_alive & at_bottom, jnp.float32(10.0), new_seg_y)
        new_seg_dx = jnp.where(
            state.seg_alive & at_bottom, jnp.float32(-_CENT_SPEED), new_seg_dx
        )

        # Bullet hits segments
        seg_hit_bx = jnp.abs(new_seg_x - new_bx) < jnp.float32(8.0)
        seg_hit_by = jnp.abs(new_seg_y - new_by) < jnp.float32(8.0)
        seg_hit = seg_hit_bx & seg_hit_by & state.seg_alive & new_bactive
        any_seg_hit = jnp.any(seg_hit)
        # Points: head (seg 0) = 100, rest = 10
        head_bonus = jnp.where(seg_hit[0], jnp.float32(90.0), jnp.float32(0.0))
        step_reward = (
            step_reward
            + jnp.float32(jnp.sum(seg_hit).astype(jnp.int32)) * jnp.float32(10.0)
            + head_bonus
        )
        new_seg_alive = state.seg_alive & ~seg_hit
        new_bactive = new_bactive & ~any_seg_hit

        # New mushroom where segment was killed
        killed_x = jnp.where(seg_hit, new_seg_x, jnp.float32(-999.0))
        killed_y = jnp.where(seg_hit, new_seg_y, jnp.float32(-999.0))
        # Place mushrooms in empty slots (simplified: update first N destroyed mushrooms)
        # This is a simplified approach — we just track overall mushroom state

        # Player–segment collision
        seg_hits_player = (
            (jnp.abs(new_seg_x - new_px) < jnp.float32(10.0))
            & (jnp.abs(new_seg_y - new_py) < jnp.float32(10.0))
            & new_seg_alive
        )
        player_killed = jnp.any(seg_hits_player)
        new_lives = state.lives - jnp.where(player_killed, jnp.int32(1), jnp.int32(0))

        # Reset on death
        new_px = jnp.where(player_killed, jnp.float32(_PLAYER_START_X), new_px)
        new_py = jnp.where(player_killed, jnp.float32(_PLAYER_START_Y), new_py)

        # Wave complete
        wave_done = ~jnp.any(new_seg_alive)
        new_seg_alive = jnp.where(
            wave_done, jnp.ones(10, dtype=jnp.bool_), new_seg_alive
        )
        new_seg_y = jnp.where(
            wave_done, jnp.full(10, 10.0, dtype=jnp.float32), new_seg_y
        )
        new_seg_x = jnp.where(
            wave_done,
            jnp.array(
                [130.0, 120.0, 110.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0],
                dtype=jnp.float32,
            ),
            new_seg_x,
        )
        new_seg_dx = jnp.where(
            wave_done, jnp.full(10, -_CENT_SPEED, dtype=jnp.float32), new_seg_dx
        )

        # Spider (roams player zone)
        new_spider_timer = state.spider_timer - jnp.int32(1)
        spider_spawn = (new_spider_timer <= jnp.int32(0)) & ~state.spider_active
        new_spider_timer = jnp.where(spider_spawn, jnp.int32(150), new_spider_timer)
        new_spider_x = jnp.where(
            state.spider_active,
            jnp.clip(
                state.spider_x
                + jax.random.uniform(k_sp) * jnp.float32(4.0)
                - jnp.float32(2.0),
                jnp.float32(5.0),
                jnp.float32(147.0),
            ),
            jnp.where(spider_spawn, jnp.float32(5.0), state.spider_x),
        )
        new_spider_y = jnp.where(
            state.spider_active,
            jnp.clip(
                state.spider_y
                + jax.random.uniform(k_sp) * jnp.float32(2.0)
                - jnp.float32(1.0),
                jnp.float32(_PLAYER_ZONE_TOP),
                jnp.float32(195.0),
            ),
            jnp.where(spider_spawn, new_py, state.spider_y),
        )
        new_spider_active = state.spider_active | spider_spawn

        # Spider exits after 200 sub-steps (approximate)
        spider_leaves = new_spider_active & (new_spider_x >= jnp.float32(150.0))
        new_spider_active = new_spider_active & ~spider_leaves
        new_spider_timer = jnp.where(spider_leaves, jnp.int32(150), new_spider_timer)

        # Spider hits player
        spider_hits_player = (
            new_spider_active
            & (jnp.abs(new_spider_x - new_px) < jnp.float32(10.0))
            & (jnp.abs(new_spider_y - new_py) < jnp.float32(10.0))
        )
        new_lives = new_lives - jnp.where(
            spider_hits_player, jnp.int32(1), jnp.int32(0)
        )
        new_spider_active = new_spider_active & ~spider_hits_player

        # Bullet hits spider
        spider_hit_by_bullet = (
            new_bactive
            & new_spider_active
            & (jnp.abs(new_spider_x - new_bx) < jnp.float32(10.0))
            & (jnp.abs(new_spider_y - new_by) < jnp.float32(10.0))
        )
        step_reward = step_reward + jnp.where(
            spider_hit_by_bullet, jnp.float32(300.0), jnp.float32(0.0)
        )
        new_spider_active = new_spider_active & ~spider_hit_by_bullet
        new_bactive = new_bactive & ~spider_hit_by_bullet

        # Flea (drops from top straight down)
        new_flea_timer = state.flea_timer - jnp.int32(1)
        flea_spawn = (new_flea_timer <= jnp.int32(0)) & ~state.flea_active
        new_flea_timer = jnp.where(flea_spawn, jnp.int32(200), new_flea_timer)
        new_flea_x = jnp.where(
            flea_spawn,
            jax.random.uniform(k_fl, minval=5.0, maxval=147.0),
            state.flea_x,
        )
        new_flea_y = jnp.where(flea_spawn, jnp.float32(0.0), state.flea_y)
        new_flea_active = state.flea_active | flea_spawn
        new_flea_y = jnp.where(new_flea_active, new_flea_y + _FLEA_SPEED, new_flea_y)
        flea_exits = new_flea_active & (new_flea_y > jnp.float32(195.0))
        new_flea_active = new_flea_active & ~flea_exits
        new_flea_timer = jnp.where(flea_exits, jnp.int32(200), new_flea_timer)

        flea_hit_by_bullet = (
            new_bactive
            & new_flea_active
            & (jnp.abs(new_flea_x - new_bx) < jnp.float32(6.0))
            & (jnp.abs(new_flea_y - new_by) < jnp.float32(6.0))
        )
        step_reward = step_reward + jnp.where(
            flea_hit_by_bullet, jnp.float32(200.0), jnp.float32(0.0)
        )
        new_flea_active = new_flea_active & ~flea_hit_by_bullet
        new_bactive = new_bactive & ~flea_hit_by_bullet

        done = new_lives <= jnp.int32(0)

        return CentipedeState(
            player_x=new_px,
            player_y=new_py,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            seg_x=new_seg_x,
            seg_y=new_seg_y,
            seg_alive=new_seg_alive,
            seg_dx=new_seg_dx,
            mush_x=state.mush_x,
            mush_y=state.mush_y,
            mush_hp=new_mush_hp,
            spider_x=new_spider_x,
            spider_y=new_spider_y,
            spider_active=new_spider_active,
            spider_timer=new_spider_timer,
            flea_x=new_flea_x,
            flea_y=new_flea_y,
            flea_active=new_flea_active,
            flea_timer=new_flea_timer,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=key,
        )

    def _step(self, state: CentipedeState, action: jax.Array) -> CentipedeState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : CentipedeState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : CentipedeState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: CentipedeState) -> CentipedeState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: CentipedeState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : CentipedeState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # Mushrooms
        def draw_mush(frm, i):
            mx = jnp.int32(state.mush_x[i])
            my = jnp.int32(state.mush_y[i])
            hp = state.mush_hp[i]
            visible = hp > jnp.int32(0)
            color = jnp.where(hp >= jnp.int32(3), _COLOR_MUSH, _COLOR_MUSH_HIT)
            mask = (
                visible
                & (_ROW_IDX >= my)
                & (_ROW_IDX < my + 6)
                & (_COL_IDX >= mx)
                & (_COL_IDX < mx + 6)
            )
            return jnp.where(mask[:, :, None], color, frm), None

        frame, _ = jax.lax.scan(draw_mush, frame, jnp.arange(_N_MUSH))

        # Centipede segments
        def draw_seg(frm, i):
            sx = jnp.int32(state.seg_x[i])
            sy = jnp.int32(state.seg_y[i])
            alive = state.seg_alive[i]
            color = jnp.where(i == jnp.int32(0), _COLOR_HEAD, _COLOR_CENTIPEDE)
            mask = (
                alive
                & (_ROW_IDX >= sy)
                & (_ROW_IDX < sy + _SEG_H)
                & (_COL_IDX >= sx)
                & (_COL_IDX < sx + _SEG_W)
            )
            return jnp.where(mask[:, :, None], color, frm), None

        frame, _ = jax.lax.scan(draw_seg, frame, jnp.arange(_N_SEG))

        # Spider
        sp_mask = (
            state.spider_active
            & (_ROW_IDX >= jnp.int32(state.spider_y))
            & (_ROW_IDX < jnp.int32(state.spider_y) + 8)
            & (_COL_IDX >= jnp.int32(state.spider_x))
            & (_COL_IDX < jnp.int32(state.spider_x) + 8)
        )
        frame = jnp.where(sp_mask[:, :, None], _COLOR_SPIDER, frame)

        # Flea
        fl_mask = (
            state.flea_active
            & (_ROW_IDX >= jnp.int32(state.flea_y))
            & (_ROW_IDX < jnp.int32(state.flea_y) + 6)
            & (_COL_IDX >= jnp.int32(state.flea_x))
            & (_COL_IDX < jnp.int32(state.flea_x) + 4)
        )
        frame = jnp.where(fl_mask[:, :, None], _COLOR_FLEA, frame)

        # Player
        pm = (
            (_ROW_IDX >= jnp.int32(state.player_y))
            & (_ROW_IDX < jnp.int32(state.player_y) + _PLAYER_H)
            & (_COL_IDX >= jnp.int32(state.player_x))
            & (_COL_IDX < jnp.int32(state.player_x) + _PLAYER_W)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        # Bullet
        bm = (
            state.bullet_active
            & (_ROW_IDX >= jnp.int32(state.bullet_y))
            & (_ROW_IDX < jnp.int32(state.bullet_y) + _BULLET_H)
            & (_COL_IDX >= jnp.int32(state.bullet_x))
            & (_COL_IDX < jnp.int32(state.bullet_x) + _BULLET_W)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Centipede action indices.
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
