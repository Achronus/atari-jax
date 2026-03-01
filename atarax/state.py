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

import chex

MAX_ENTITIES: int = 64
"""Enemies, obstacles, neutral objects."""
MAX_PROJECTILES: int = 32
"""Player + enemy bullets."""


@chex.dataclass
class GameState:
    """
    Base dataclass for all JAX game state structs.

    Every game's concrete state class must inherit from `GameState` and declare
    these four fields (plus any game-specific fields).

    Parameters
    ----------
    reward : chex.Array
        float32 — Reward earned on the last `step` call.
    done : chex.Array
        bool — `True` when the episode has ended (game over or truncated).
    step : chex.Array
        int32 — Total emulated steps since power-on.
    episode_step : chex.Array
        int32 — Agent steps elapsed in the current episode.
    """

    reward: chex.Array
    done: chex.Array
    step: chex.Array
    episode_step: chex.Array


@chex.dataclass
class AtariState(GameState):
    """
    Atari-specific extension of `GameState`.

    Adds fields common to all Atari 2600 games. Game-specific state
    dataclasses (e.g. `BreakoutState`) inherit from `AtariState` and add
    their own fields.

    Parameters
    ----------
    lives : chex.Array
        int32 — Remaining lives (use `0` for games without lives).
    score : chex.Array
        int32 — Cumulative episode score.
    level : chex.Array
        int32 — Board / wave / level counter; `0` = first board.
    key : chex.PRNGKey
        JAX PRNG key, evolved each physics sub-step.
    """

    lives: chex.Array
    score: chex.Array
    level: chex.Array
    key: chex.PRNGKey


@chex.dataclass
class AtaraxState:
    """
    Unified pytree state schema for all 57 Atari games

    All 57 games share this single concrete array shape (Hard-tier
    capacities), enabling `jax.vmap` across all games simultaneously.
    Unused fields are zeroed — inactive entities are masked, not branched.

    Parameters
    ----------
    rng_key : chex.Array
        (2,) uint32 — JAX PRNG key.
    player_x : chex.Array
        float32 — Player x position in native ALE coordinates (0–159).
    player_y : chex.Array
        float32 — Player y position in native ALE coordinates (0–191).
    player_vx : chex.Array
        float32 — Player x velocity; zero for discrete-movement games.
    player_vy : chex.Array
        float32 — Player y velocity.
    player_alive : chex.Array
        bool — `False` triggers terminal / respawn.
    score : chex.Array
        int32 — Cumulative episode score.
    lives : chex.Array
        int32 — Remaining lives; `1` for no-lives games.
    step_count : chex.Array
        int32 — Frames since episode start.
    terminal : chex.Array
        bool — Set by step_fn, consumed by rollout wrapper.
    entities_x : chex.Array
        (MAX_ENTITIES,) float32 — Entity x positions; inactive slots are masked.
    entities_y : chex.Array
        (MAX_ENTITIES,) float32 — Entity y positions.
    entities_vx : chex.Array
        (MAX_ENTITIES,) float32 — Entity x velocities.
    entities_vy : chex.Array
        (MAX_ENTITIES,) float32 — Entity y velocities.
    entities_type : chex.Array
        (MAX_ENTITIES,) int32 — Per-game entity type enum.
    entities_active : chex.Array
        (MAX_ENTITIES,) bool — Active mask; inactive slots are ignored.
    projectiles_x : chex.Array
        (MAX_PROJECTILES,) float32 — Projectile x positions.
    projectiles_y : chex.Array
        (MAX_PROJECTILES,) float32 — Projectile y positions.
    projectiles_vx : chex.Array
        (MAX_PROJECTILES,) float32 — Projectile x velocities.
    projectiles_vy : chex.Array
        (MAX_PROJECTILES,) float32 — Projectile y velocities.
    projectiles_owner : chex.Array
        (MAX_PROJECTILES,) int32 — Owner: `0` = player, `1` = enemy.
    projectiles_active : chex.Array
        (MAX_PROJECTILES,) bool — Active mask.
    grid : chex.Array
        (32, 32) bool — Destructible / static tile map; zeroed in Easy games.
    grid_values : chex.Array
        (32, 32) int32 — Per-tile multi-hit count or type; zeroed in Easy games.
    timer_a : chex.Array
        int32 — General-purpose timer A (respawn delay, powerup duration, etc.).
    timer_b : chex.Array
        int32 — General-purpose timer B (second independent timer).
    difficulty_level : chex.Array
        int32 — Governs enemy speed / spawn rate.
    wave_level : chex.Array
        int32 — Current wave / level index.
    enemy_dir : chex.Array
        int32 — Space Invaders fleet direction.
    ball_x : chex.Array
        float32 — Ball x position (Breakout, Tennis, Pong, Video Pinball).
    ball_y : chex.Array
        float32 — Ball y position.
    ball_vx : chex.Array
        float32 — Ball x velocity.
    ball_vy : chex.Array
        float32 — Ball y velocity.
    paddle_y : chex.Array
        float32 — Player paddle y position.
    paddle2_y : chex.Array
        float32 — Opponent paddle y position (Pong, Tennis, Fishing Derby).
    resources : chex.Array
        (8,) int32 — Oxygen / fuel / ammo / keys / etc.
    room_id : chex.Array
        int32 — Indexes into the static per-game room table; zeroed in Easy/Medium games.
    room_grid : chex.Array
        (32, 32) int32 — Current room tile map; zeroed in Easy/Medium games.
    visited_rooms : chex.Array
        (256,) bool — Exploration mask; zeroed in Easy/Medium games.
    inventory : chex.Array
        (16,) int32 — Keys / sword / torch / etc.; zeroed in Easy/Medium games.
    physics_angle : chex.Array
        float32 — Player rotation in radians; zeroed in Easy/Medium games.
    physics_thrust : chex.Array
        float32 — Current thrust magnitude; zeroed in Easy/Medium games.
    scroll_offset : chex.Array
        float32 — Scroll offset for Pitfall, Enduro, Road Runner, Skiing; zeroed in Easy/Medium games.
    credit_timer : chex.Array
        int32 — Skiing gate penalty / reward-at-end-of-run; zeroed in Easy/Medium games.
    """

    # PRNG key
    rng_key: chex.Array

    # Player
    player_x: chex.Array
    player_y: chex.Array
    player_vx: chex.Array
    player_vy: chex.Array
    player_alive: chex.Array

    # Episode meta
    score: chex.Array
    lives: chex.Array
    step_count: chex.Array
    terminal: chex.Array

    # Entity table — MAX_ENTITIES slots; inactive slots are masked, not branched
    entities_x: chex.Array
    entities_y: chex.Array
    entities_vx: chex.Array
    entities_vy: chex.Array
    entities_type: chex.Array
    entities_active: chex.Array

    # Projectile table — MAX_PROJECTILES slots
    projectiles_x: chex.Array
    projectiles_y: chex.Array
    projectiles_vx: chex.Array
    projectiles_vy: chex.Array
    projectiles_owner: chex.Array
    projectiles_active: chex.Array

    # Medium extensions (zeroed/false in Easy games)
    grid: chex.Array
    grid_values: chex.Array
    timer_a: chex.Array
    timer_b: chex.Array
    difficulty_level: chex.Array
    wave_level: chex.Array
    enemy_dir: chex.Array
    ball_x: chex.Array
    ball_y: chex.Array
    ball_vx: chex.Array
    ball_vy: chex.Array
    paddle_y: chex.Array
    paddle2_y: chex.Array
    resources: chex.Array

    # Hard extensions (zeroed in Easy/Medium games)
    room_id: chex.Array
    room_grid: chex.Array
    visited_rooms: chex.Array
    inventory: chex.Array
    physics_angle: chex.Array
    physics_thrust: chex.Array
    scroll_offset: chex.Array
    credit_timer: chex.Array
