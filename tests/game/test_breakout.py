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

"""Unit tests for atarax.games.breakout — BreakoutState and game logic.

Run with:
    pytest tests/game/test_breakout.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atarax.games._base import AtariState
from atarax.games.breakout import Breakout, BreakoutState

_key = jax.random.PRNGKey(0)

game = Breakout()


def test_reset_returns_breakout_state():
    _, state = game.reset(_key)
    assert isinstance(state, BreakoutState)
    assert isinstance(state, AtariState)


def test_reset_bricks_shape():
    _, state = game.reset(_key)
    chex.assert_shape(state.bricks, (6, 18))


def test_reset_all_bricks_active():
    _, state = game.reset(_key)
    assert bool(jnp.all(state.bricks))


def test_reset_lives():
    _, state = game.reset(_key)
    chex.assert_rank(state.lives, 0)
    assert int(state.lives) == 5


def test_reset_score_zero():
    _, state = game.reset(_key)
    assert int(state.score) == 0


def test_reset_done_false():
    _, state = game.reset(_key)
    assert not bool(state.done)


def test_reset_reward_zero():
    _, state = game.reset(_key)
    assert float(state.reward) == 0.0


def test_reset_deterministic():
    _, s1 = game.reset(jax.random.PRNGKey(42))
    _, s2 = game.reset(jax.random.PRNGKey(42))
    assert bool(jnp.all(s1.bricks == s2.bricks))
    assert float(s1.ball_x) == float(s2.ball_x)
    assert float(s1.paddle_x) == float(s2.paddle_x)


def test_reset_different_keys_same_bricks():
    _, s1 = game.reset(jax.random.PRNGKey(0))
    _, s2 = game.reset(jax.random.PRNGKey(99))
    assert bool(jnp.all(s1.bricks == s2.bricks))


def test_step_returns_breakout_state():
    _, state = game.reset(_key)
    _, new_state, _, _, _ = game.step(state, jnp.int32(0))
    assert isinstance(new_state, BreakoutState)


def test_step_reward_type():
    _, state = game.reset(_key)
    _, new_state, reward, _, _ = game.step(state, jnp.int32(0))
    chex.assert_rank(reward, 0)
    chex.assert_type(reward, jnp.float32)


def test_step_done_type():
    _, state = game.reset(_key)
    _, _, _, done, _ = game.step(state, jnp.int32(0))
    chex.assert_rank(done, 0)
    chex.assert_type(done, jnp.bool_)


def test_step_episode_step_increments():
    _, state = game.reset(_key)
    _, new_state, _, _, _ = game.step(state, jnp.int32(0))
    assert int(new_state.episode_step) > int(state.episode_step)


def test_render_shape():
    _, state = game.reset(_key)
    frame = game.render(state)
    chex.assert_shape(frame, (210, 160, 3))
    chex.assert_type(frame, jnp.uint8)


def test_vmap_reset():
    keys = jax.random.split(_key, 8)
    obs_batch, states = jax.vmap(game.reset)(keys)
    chex.assert_shape(states.bricks, (8, 6, 18))
    chex.assert_shape(states.ball_x, (8,))


def test_jit_step():
    _, state = game.reset(_key)
    step_jit = jax.jit(game.step)
    _, new_state, _, _, _ = step_jit(state, jnp.int32(0))
    assert isinstance(new_state, BreakoutState)


def test_pytree_flatten():
    _, state = game.reset(_key)
    flat, treedef = jax.tree_util.tree_flatten(state)
    assert len(flat) > 0
    restored = jax.tree_util.tree_unflatten(treedef, flat)
    assert bool(jnp.all(restored.bricks == state.bricks))
