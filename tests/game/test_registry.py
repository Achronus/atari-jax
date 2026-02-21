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

"""Unit tests for the game registry and jax.lax.switch dispatch.

Run with:
    pytest tests/game/test_registry.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atarax.games import GAME_IDS, get_reward, is_terminal
from atarax.games.registry import _GAMES, REWARD_FNS, TERMINAL_FNS


def test_game_ids_contains_breakout():
    assert "breakout" in GAME_IDS
    assert GAME_IDS["breakout"] == 12


def test_games_list_indexed_by_game_id():
    for spec in _GAMES:
        assert _GAMES[spec.game_id].ale_name == spec.ale_name


def test_registry_has_57_games():
    assert len(_GAMES) == 57


def test_reward_fns_has_57_entries():
    assert len(REWARD_FNS) == 57


def test_terminal_fns_has_57_entries():
    assert len(TERMINAL_FNS) == 57


def test_game_ids_dict_has_57_entries():
    assert len(GAME_IDS) == 57


def test_all_ale_names_unique():
    names = [spec.ale_name for spec in _GAMES]
    assert len(names) == len(set(names))


def test_reward_fns_are_callable():
    assert all(callable(fn) for fn in REWARD_FNS)


def test_terminal_fns_are_callable():
    assert all(callable(fn) for fn in TERMINAL_FNS)


def _zeros_ram() -> chex.Array:
    return jnp.zeros(128, dtype=jnp.uint8)


def test_dispatch_get_reward_compiles():
    """get_reward dispatch via jax.lax.switch should JIT-compile."""
    ram = _zeros_ram()
    game_id = jnp.int32(GAME_IDS["breakout"])
    result = jax.jit(get_reward)(game_id, ram, ram)
    chex.assert_rank(result, 0)
    chex.assert_type(result, jnp.float32)
    assert float(result) == 0.0


def test_dispatch_is_terminal_compiles():
    """is_terminal dispatch via jax.lax.switch should JIT-compile."""
    ram = _zeros_ram()
    game_id = jnp.int32(GAME_IDS["breakout"])
    lives_prev = jnp.int32(0)  # game never started → always non-terminal
    result = jax.jit(is_terminal)(game_id, ram, lives_prev)
    chex.assert_rank(result, 0)
    chex.assert_type(result, bool)
    assert not bool(result)


def test_dispatch_get_reward_vmap():
    """get_reward should be vmappable over a batch of game_ids."""
    n = 4
    ram = _zeros_ram()
    game_ids = jnp.zeros(n, dtype=jnp.int32)
    rams = jnp.stack([ram] * n)
    results = jax.vmap(lambda gid, r: get_reward(gid, r, r))(game_ids, rams)
    chex.assert_shape(results, (n,))
    chex.assert_type(results, jnp.float32)
