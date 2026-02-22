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

import chex
import jax

from atarax.games.registry import GAME_IDS, SCORE_FNS, TERMINAL_FNS

__all__ = ["GAME_IDS", "get_score", "is_terminal"]


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
