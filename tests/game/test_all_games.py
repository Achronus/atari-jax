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

"""Parametrised smoke-tests covering every registered Atari-JAX game.

Each test verifies that a game satisfies the ``AtaraxGame`` contract without
testing game-specific mechanics.  Tests are parameterised over all entries
in the ``GAMES`` registry so adding a new game automatically includes it.

Run all games::

    pytest tests/game/test_all_games.py -v

Run a single game::

    pytest tests/game/test_all_games.py -v -k breakout
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax.game import AtaraxParams
from atarax.games import GAMES
from atarax.state import AtariState

_KEY = jax.random.PRNGKey(0)
_PARAMS = AtaraxParams(noop_max=0)  # disable NOOP starts in tests for reproducibility


@pytest.fixture(scope="module", params=sorted(GAMES.keys()))
def game(request):
    return GAMES[request.param]()


def test_reset(game):
    _, state = game.reset(_KEY, _PARAMS)
    assert isinstance(state, AtariState)
    assert int(state.score) == 0
    assert not bool(state.done)
    assert float(state.reward) == 0.0
    chex.assert_rank(state.lives, 0)
    chex.assert_rank(state.score, 0)
    chex.assert_rank(state.episode_step, 0)
    chex.assert_rank(state.level, 0)
    assert state.level.dtype == jnp.int32


def test_reset_deterministic(game):
    _, s1 = game.reset(jax.random.PRNGKey(7), _PARAMS)
    _, s2 = game.reset(jax.random.PRNGKey(7), _PARAMS)
    assert int(s1.score) == int(s2.score)
    assert bool(s1.done) == bool(s2.done)


def test_step(game):
    _, state = game.reset(_KEY, _PARAMS)
    _, new_state, reward, done, info = game.step(_KEY, state, jnp.int32(0), _PARAMS)
    assert isinstance(new_state, AtariState)
    chex.assert_rank(reward, 0)
    assert reward.dtype == jnp.float32
    chex.assert_rank(done, 0)
    assert done.dtype == jnp.bool_
    assert int(new_state.episode_step) > int(state.episode_step)
    assert set(info.keys()) >= {"lives", "score", "level", "episode_step", "truncated"}


def test_render(game):
    _, state = game.reset(_KEY, _PARAMS)
    frame = game.render(state)
    chex.assert_shape(frame, (210, 160, 3))
    assert frame.dtype == jnp.uint8


def test_vmap(game):
    keys = jax.random.split(_KEY, 4)
    obs_batch, states = jax.vmap(game.reset, in_axes=(0, None))(keys, _PARAMS)
    chex.assert_shape(obs_batch, (4, 210, 160, 3))
    chex.assert_shape(states.score, (4,))
    chex.assert_shape(states.lives, (4,))


def test_jit(game):
    _, state = game.reset(_KEY, _PARAMS)
    step_fn = jax.jit(game.step)
    _, new_state, reward, done, _ = step_fn(_KEY, state, jnp.int32(0), _PARAMS)
    assert isinstance(new_state, AtariState)
    assert reward.dtype == jnp.float32
    assert done.dtype == jnp.bool_


def test_pytree(game):
    _, state = game.reset(_KEY, _PARAMS)
    flat, treedef = jax.tree_util.tree_flatten(state)
    assert len(flat) > 0
    restored = jax.tree_util.tree_unflatten(treedef, flat)
    assert int(restored.score) == int(state.score)
    chex.assert_rank(restored.score, 0)
