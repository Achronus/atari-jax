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

"""Unit tests for the shared kernel dispatch functions and _jit_sample.

ROM-free: only the dispatch functions (get_lives, compute_reward_and_score)
and the sample kernel are tested here — these operate purely on RAM arrays
and game IDs without needing actual ROM bytes.

Full integration tests for jit_reset / jit_step / jit_vec_* are covered
by tests/env/test_make.py and tests/env/test_vec_env.py.

Run with:
    pytest tests/env/test_kernels.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env._kernels import _jit_sample
from atarax.games import compute_reward_and_score, get_lives
from atarax.games.registry import GAME_IDS

# Use Breakout (score-tracking) and Tennis (direct-reward) as representative games.
_BREAKOUT_ID = jnp.int32(GAME_IDS["breakout"])
_TENNIS_ID = jnp.int32(GAME_IDS["tennis"])
_ZERO_RAM = jnp.zeros(128, dtype=jnp.uint8)


def test_get_lives_returns_int32():
    lives = get_lives(_BREAKOUT_ID, _ZERO_RAM)
    chex.assert_rank(lives, 0)
    chex.assert_type(lives, jnp.int32)


def test_get_lives_breakout_zero_ram():
    # Breakout lives address (57) is 0 in zero RAM → 0 lives.
    lives = get_lives(_BREAKOUT_ID, _ZERO_RAM)
    assert int(lives) == 0


def test_get_lives_tennis_always_zero():
    # Tennis has no lives counter; always returns 0.
    lives = get_lives(_TENNIS_ID, _ZERO_RAM)
    assert int(lives) == 0


def test_compute_reward_and_score_tracking_no_change():
    # Breakout: score delta = 0 when RAM unchanged.
    reward, new_score = compute_reward_and_score(
        _BREAKOUT_ID, _ZERO_RAM, _ZERO_RAM, jnp.int32(0)
    )
    chex.assert_rank(reward, 0)
    chex.assert_type(reward, jnp.float32)
    chex.assert_rank(new_score, 0)
    chex.assert_type(new_score, jnp.int32)
    assert float(reward) == 0.0
    assert int(new_score) == 0


def test_compute_reward_and_score_tracking_prev_score_carried():
    # Breakout: new_score reflects current RAM score regardless of prev_score.
    # With zero RAM, get_score(zero_ram) = 0 → new_score = 0, reward = 0 - 5 = -5.
    reward, new_score = compute_reward_and_score(
        _BREAKOUT_ID, _ZERO_RAM, _ZERO_RAM, jnp.int32(5)
    )
    assert float(reward) == -5.0
    assert int(new_score) == 0


def test_compute_reward_and_score_tennis_no_change():
    # Tennis (direct-reward): no RAM change → reward = 0, prev_score unchanged.
    reward, new_score = compute_reward_and_score(
        _TENNIS_ID, _ZERO_RAM, _ZERO_RAM, jnp.int32(99)
    )
    chex.assert_type(reward, jnp.float32)
    assert float(reward) == 0.0
    assert int(new_score) == 99  # score passthrough for non-tracking games


def test_jit_sample_shape_and_dtype():
    key = jax.random.PRNGKey(0)
    action = _jit_sample(key)
    chex.assert_rank(action, 0)
    chex.assert_type(action, jnp.int32)
    assert 0 <= int(action) < 18


def test_jit_sample_different_keys():
    # Different keys should produce at least some variation across many samples.
    keys = jax.random.split(jax.random.PRNGKey(42), 100)
    actions = jax.vmap(_jit_sample)(keys)
    assert len(set(actions.tolist())) > 1
