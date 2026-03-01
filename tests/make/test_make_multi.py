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

"""Tests for make_multi() and make_multi_vec().

Run with:
    pytest tests/make/test_make_multi.py -v
"""

import chex
import jax

from atarax import VmapEnv, make_multi, make_multi_vec
from atarax.game import AtaraxGame
from atarax.spec import EnvSpec

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"
_BREAKOUT_SPEC = EnvSpec("atari", "breakout")
_IDS = [_BREAKOUT, _BREAKOUT_SPEC]   # one str, one EnvSpec
_N_ENVS = 2


def test_make_multi_returns_list():
    results = make_multi([_BREAKOUT], jit_compile=False)
    assert isinstance(results, list)


def test_make_multi_length():
    results = make_multi(_IDS, jit_compile=False)
    assert len(results) == len(_IDS)


def test_make_multi_all_are_atarax_game():
    results = make_multi(_IDS, jit_compile=False)
    assert all(isinstance(e, AtaraxGame) for e, _ in results)


def test_make_multi_reset_shape():
    results = make_multi([_BREAKOUT], jit_compile=False)
    env, params = results[0]
    obs, _ = env.reset(_key, params)
    chex.assert_shape(obs, (210, 160, 3))


def test_make_multi_accepts_env_spec():
    results = make_multi([_BREAKOUT_SPEC], jit_compile=False)
    assert len(results) == 1
    assert isinstance(results[0][0], AtaraxGame)


def test_make_multi_preset():
    results = make_multi([_BREAKOUT], preset=True, jit_compile=False)
    env, params = results[0]
    obs, _ = env.reset(_key, params)
    assert obs.shape == (84, 84, 4)


def test_make_multi_vec_returns_list():
    results = make_multi_vec([_BREAKOUT], _N_ENVS, jit_compile=False)
    assert isinstance(results, list)


def test_make_multi_vec_length():
    results = make_multi_vec(_IDS, _N_ENVS, jit_compile=False)
    assert len(results) == len(_IDS)


def test_make_multi_vec_all_are_vmap_env():
    results = make_multi_vec(_IDS, _N_ENVS, jit_compile=False)
    assert all(isinstance(v, VmapEnv) for v, _ in results)


def test_make_multi_vec_num_envs():
    results = make_multi_vec([_BREAKOUT], _N_ENVS, jit_compile=False)
    vec_env, _ = results[0]
    assert vec_env.num_envs == _N_ENVS


def test_make_multi_vec_reset_shape():
    results = make_multi_vec([_BREAKOUT], _N_ENVS, jit_compile=False)
    vec_env, params = results[0]
    obs, _ = vec_env.reset(_key, params)
    chex.assert_shape(obs, (_N_ENVS, 210, 160, 3))


def test_make_multi_vec_step_shape():
    import jax.numpy as jnp
    results = make_multi_vec([_BREAKOUT], _N_ENVS, jit_compile=False)
    vec_env, params = results[0]
    _, states = vec_env.reset(_key, params)
    actions = jnp.zeros(_N_ENVS, dtype=jnp.int32)
    obs, _, reward, done, _ = vec_env.step(_key, states, actions, params)
    chex.assert_shape(obs, (_N_ENVS, 210, 160, 3))
    chex.assert_shape(reward, (_N_ENVS,))


def test_make_multi_vec_accepts_env_spec():
    results = make_multi_vec([_BREAKOUT_SPEC], _N_ENVS, jit_compile=False)
    assert len(results) == 1
    assert isinstance(results[0][0], VmapEnv)
