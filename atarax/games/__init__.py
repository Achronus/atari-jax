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

"""Per-game score and terminal extraction, dispatched via jax.lax.switch."""

from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp

from atarax.games.registry import (
    GAME_IDS,
    LIVES_FNS,
    REWARD_SCORE_FNS,
    SCORE_FNS,
    TERMINAL_FNS,
)

__all__ = [
    "GAME_IDS",
    "build_group_dispatch",
    "compute_reward_and_score",
    "get_lives",
    "get_score",
    "is_terminal",
]


def get_score(
    game_id: chex.Array,
    ram: chex.Array,
) -> chex.Array:
    """
    Dispatch raw score extraction to the appropriate game implementation.

    Uses `jax.lax.switch` so the call is fully JAX-traceable and compatible
    with `jax.jit` and `jax.vmap`.

    Parameters
    ----------
    game_id : chex.Array
        int32 — Index into the game registry (see `GAME_IDS`).
    ram : chex.Array
        uint8[128] — RIOT RAM snapshot.

    Returns
    -------
    score : chex.Array
        int32 — Raw game score (use `state.score` delta for reward).
    """
    return jax.lax.switch(game_id, SCORE_FNS, ram)


def is_terminal(
    game_id: chex.Array,
    ram: chex.Array,
    lives_prev: chex.Array,
) -> chex.Array:
    """
    Dispatch terminal detection to the appropriate game implementation.

    Uses `jax.lax.switch` so the call is fully JAX-traceable and compatible
    with `jax.jit` and `jax.vmap`.

    Parameters
    ----------
    game_id : chex.Array
        int32 — Index into the game registry (see `GAME_IDS`).
    ram : chex.Array
        uint8[128] — RIOT RAM after the step.
    lives_prev : chex.Array
        int32 — Lives count before the step.

    Returns
    -------
    terminal : chex.Array
        bool — True when the episode ended on this step.
    """
    return jax.lax.switch(game_id, TERMINAL_FNS, ram, lives_prev)


def get_lives(
    game_id: chex.Array,
    ram: chex.Array,
) -> chex.Array:
    """
    Dispatch lives-count extraction to the appropriate game implementation.

    Uses `jax.lax.switch` so the call is fully JAX-traceable and compatible
    with `jax.jit` and `jax.vmap`.

    Parameters
    ----------
    game_id : chex.Array
        int32 — Index into the game registry (see `GAME_IDS`).
    ram : chex.Array
        uint8[128] — RIOT RAM snapshot.

    Returns
    -------
    lives : chex.Array
        int32 — Lives remaining.
    """
    return jax.lax.switch(game_id, LIVES_FNS, ram)


def compute_reward_and_score(
    game_id: chex.Array,
    ram_prev: chex.Array,
    ram_curr: chex.Array,
    prev_score: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """
    Dispatch reward and score computation to the appropriate game implementation.

    Handles both score-tracking games (reward = score delta) and direct-reward
    games like Tennis (reward from two RAM snapshots, score unchanged).

    Uses `jax.lax.switch` so the call is fully JAX-traceable and compatible
    with `jax.jit` and `jax.vmap`.

    Parameters
    ----------
    game_id : chex.Array
        int32 — Index into the game registry (see `GAME_IDS`).
    ram_prev : chex.Array
        uint8[128] — RIOT RAM before the step.
    ram_curr : chex.Array
        uint8[128] — RIOT RAM after the step.
    prev_score : chex.Array
        int32 — Score accumulated before this step.

    Returns
    -------
    reward : chex.Array
        float32 — Reward earned on this step.
    new_score : chex.Array
        int32 — Updated cumulative score.
    """
    return jax.lax.switch(game_id, REWARD_SCORE_FNS, ram_prev, ram_curr, prev_score)


def build_group_dispatch(
    group_game_ids: Tuple[int, ...],
) -> Tuple[Callable, Callable, Callable]:
    """
    Build compact N-way dispatch functions for a subset of games.

    Slices the global per-game function tables to only the N games in
    `group_game_ids` and returns three dispatch callables that use an
    N-way `jax.lax.switch` instead of the full 57-way switch.  The
    absolute-game-id → group-index remapping is handled via a lookup into
    a `(57,)` mapping array baked into each closure.

    Parameters
    ----------
    group_game_ids : tuple of int
        Sorted tuple of absolute game IDs (values from `GAME_IDS`) that
        form the group.  Must be non-empty and contain only valid IDs.

    Returns
    -------
    compute_reward_and_score_g : Callable
        Group-scoped reward and score dispatch.
    get_lives_g : Callable
        Group-scoped lives dispatch.
    is_terminal_g : Callable
        Group-scoped terminal dispatch.
    """
    n_all = len(GAME_IDS)
    group_map = jnp.full(n_all, -1, dtype=jnp.int32)
    for group_idx, abs_id in enumerate(group_game_ids):
        group_map = group_map.at[abs_id].set(group_idx)

    group_reward_fns = [REWARD_SCORE_FNS[i] for i in group_game_ids]
    group_lives_fns = [LIVES_FNS[i] for i in group_game_ids]
    group_terminal_fns = [TERMINAL_FNS[i] for i in group_game_ids]

    def compute_reward_and_score_g(
        game_id: chex.Array,
        ram_prev: chex.Array,
        ram_curr: chex.Array,
        prev_score: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        return jax.lax.switch(
            group_map[game_id], group_reward_fns, ram_prev, ram_curr, prev_score
        )

    def get_lives_g(game_id: chex.Array, ram: chex.Array) -> chex.Array:
        return jax.lax.switch(group_map[game_id], group_lives_fns, ram)

    def is_terminal_g(
        game_id: chex.Array, ram: chex.Array, lives_prev: chex.Array
    ) -> chex.Array:
        return jax.lax.switch(group_map[game_id], group_terminal_fns, ram, lives_prev)

    return compute_reward_and_score_g, get_lives_g, is_terminal_g
