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

"""Per-game reward and terminal extraction, dispatched via jax.lax.switch."""

import chex
import jax

from atari_jax.games.registry import GAME_IDS, REWARD_FNS, TERMINAL_FNS

__all__ = ["GAME_IDS", "get_reward", "is_terminal"]


def get_reward(
    game_id: chex.Array,
    ram_prev: chex.Array,
    ram_curr: chex.Array,
) -> chex.Array:
    """
    Dispatch reward computation to the appropriate game implementation.

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

    Returns
    -------
    reward : chex.Array
        float32 — Score gained this step.
    """
    return jax.lax.switch(game_id, REWARD_FNS, ram_prev, ram_curr)


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
