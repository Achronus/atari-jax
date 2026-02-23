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

"""Tests for the make_multi() factory function.

ROM-backed: requires ale-py to load game ROMs.

Run with:
    pytest tests/make/test_make_multi.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax.env import Env, make_multi
from atarax.env.atari_env import AtariEnv

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"
_MS_PACMAN = "atari/ms_pacman-v0"


def test_make_multi_returns_list_of_env():
    envs = make_multi([_BREAKOUT, _MS_PACMAN], jit_compile=False)
    assert isinstance(envs, list)
    assert len(envs) == 2
    assert all(isinstance(e, Env) for e in envs)


def test_make_multi_group_mode():
    envs = make_multi([_BREAKOUT, _MS_PACMAN], jit_compile=False)
    for env in envs:
        assert isinstance(env, AtariEnv)
        assert env._compile_mode == "group"


def test_make_multi_reset_shape():
    envs = make_multi([_BREAKOUT, _MS_PACMAN], jit_compile=False)
    obs, _ = envs[0].reset(_key)
    chex.assert_shape(obs, (210, 160, 3))


def test_make_multi_step_shape():
    envs = make_multi([_BREAKOUT, _MS_PACMAN], jit_compile=False)
    _, state = envs[0].reset(_key)
    obs, _, reward, done, _ = envs[0].step(state, jnp.int32(0))
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)


def test_make_multi_predefined_group():
    from atarax.games.registry import GAME_GROUPS

    group_names = [f"atari/{n}-v0" for n in GAME_GROUPS["atari5"]]
    envs = make_multi(group_names, jit_compile=False)
    assert len(envs) == len(group_names)
    assert all(isinstance(e, Env) for e in envs)


def test_make_multi_kernels_shared():
    envs = make_multi([_BREAKOUT, _MS_PACMAN], jit_compile=False)
    assert envs[0]._group_kernels is envs[1]._group_kernels


def test_make_multi_raises_on_empty():
    with pytest.raises(ValueError, match="game_ids"):
        make_multi([])
