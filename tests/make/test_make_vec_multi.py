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

"""Tests for the make_vec_multi() factory function.

ROM-backed: requires ale-py to load game ROMs.

Run with:
    pytest tests/make/test_make_vec_multi.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax.env import VecEnv, make_vec_multi

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"
_MS_PACMAN = "atari/ms_pacman-v0"


def test_make_vec_multi_returns_list_of_vec_env():
    vec_envs = make_vec_multi([_BREAKOUT, _MS_PACMAN], n_envs=2, jit_compile=False)
    assert isinstance(vec_envs, list)
    assert len(vec_envs) == 2
    assert all(isinstance(ve, VecEnv) for ve in vec_envs)


def test_make_vec_multi_reset_shape():
    vec_envs = make_vec_multi([_BREAKOUT, _MS_PACMAN], n_envs=2, jit_compile=False)
    obs, _ = vec_envs[0].reset(_key)
    chex.assert_shape(obs, (2, 210, 160, 3))


def test_make_vec_multi_step_shape():
    vec_envs = make_vec_multi([_BREAKOUT, _MS_PACMAN], n_envs=2, jit_compile=False)
    _, states = vec_envs[0].reset(_key)
    obs, _, reward, done, _ = vec_envs[0].step(states, jnp.zeros(2, dtype=jnp.int32))
    chex.assert_shape(obs, (2, 210, 160, 3))
    chex.assert_shape(reward, (2,))


def test_make_vec_multi_raises_on_empty():
    with pytest.raises(ValueError, match="game_ids"):
        make_vec_multi([], n_envs=2)
