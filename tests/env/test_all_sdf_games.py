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

"""Smoke tests that run against every registered SDF game.

Parametrised over ``GAMES`` so new games are automatically picked up when
added to ``atarax/env/registry.py``.

Run with::

    pytest tests/env/test_all_sdf_games.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atarax.env.registry import GAMES, PARAMS

_KEY = jax.random.PRNGKey(0)
_GAME_IDS = sorted(GAMES.keys())


@pytest.fixture(scope="module", params=_GAME_IDS)
def game_and_params(request):
    """(env, params) for each registered game."""
    gid = request.param
    env = GAMES[gid]()
    params = PARAMS[gid]()
    return env, params


def test_reset(game_and_params):
    env, params = game_and_params
    # noop_max=0 gives a clean initial state without stochastic NOOP steps.
    clean_params = params.__replace__(noop_max=0)
    obs, state = env.reset(_KEY, clean_params)
    assert obs.shape == (210, 160, 3)
    assert obs.dtype == np.uint8
    assert float(state.score) == 0.0
    assert not bool(state.done)
    assert int(state.episode_step) == 0


def test_step(game_and_params):
    env, params = game_and_params
    _, state = env.reset(_KEY, params)
    obs2, state2, reward, done, info = env.step(_KEY, state, jnp.int32(0), params)
    assert obs2.shape == (210, 160, 3)
    assert obs2.dtype == np.uint8
    assert int(state2.episode_step) > 0
    assert "lives" in info
    assert "episode_step" in info


def test_render(game_and_params):
    env, params = game_and_params
    _, state = env.reset(_KEY, params)
    frame = env.render(state)
    assert frame.shape == (210, 160, 3)
    assert frame.dtype == np.uint8


def test_jit(game_and_params):
    env, params = game_and_params
    reset_fn = jax.jit(lambda k: env.reset(k, params))
    step_fn = jax.jit(lambda k, s, a: env.step(k, s, a, params))
    obs, state = reset_fn(_KEY)
    obs2, state2, r, done, info = step_fn(_KEY, state, jnp.int32(0))
    assert obs2.shape == (210, 160, 3)


def test_vmap(game_and_params):
    env, params = game_and_params
    keys = jax.random.split(_KEY, 8)
    obs_batch, states = jax.vmap(lambda k: env.reset(k, params))(keys)
    assert obs_batch.shape == (8, 210, 160, 3)


def test_pytree(game_and_params):
    env, params = game_and_params
    _, state = env.reset(_KEY, params)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    state2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert int(state2.episode_step) == int(state.episode_step)
