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

"""ALE fidelity regression tests for Track 2 SDF games.

Runs N=1000 random-policy episodes for each game (via calibrate_fidelity.py
methodology) and checks the mean return falls within a pre-measured band.

These tests are slow (~minutes per game). Mark with ``pytest -m slow`` to
run selectively.

Calibration bands are added here as each game is measured.

Run all::

    pytest tests/env/test_ale_fidelity.py -m slow -v

Run one game::

    pytest tests/env/test_ale_fidelity.py -m slow -k space_invaders -v
"""

import numpy as np
import pytest

pytest.importorskip("ale_py", reason="ale_py required for fidelity tests")

import ale_py  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from atarax.env.registry import GAMES, PARAMS  # noqa: E402

# ---------------------------------------------------------------------------
# Protocol constants (must match calibrate_fidelity.py)
# ---------------------------------------------------------------------------
_N_ENVS = 1000
_MAX_STEPS = 1000
_SEED = 42

# ---------------------------------------------------------------------------
# ALE fidelity bands (lo, hi) — mean return must land within this range.
# Bands added as each game is calibrated.
# ---------------------------------------------------------------------------
_BANDS: dict[str, tuple[float, float]] = {
    # game_id: (lo, hi) — from scripts/calibrate_fidelity.py output (N=1000, SEED=42)
    # Breakout: accepted 9.13× deviation — larger ball + random policy geometric catch rate vs ALE ROM physics
    "asteroids": (714.7, 800.1),
    "breakout": (9.6, 10.4),
    "ms_pacman": (258.9, 269.2),
    "space_invaders": (148.1, 163.0),
}


# ---------------------------------------------------------------------------
# JAX random-policy runner (vmapped)
# ---------------------------------------------------------------------------
def _run_jax(game_id: str, n_envs: int, max_steps: int, seed: int) -> float:
    game = GAMES[game_id]()
    params = PARAMS[game_id]()
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)

    _, init_states = jax.vmap(game.reset, in_axes=(0, None))(keys, params)

    init_carry = (
        keys,
        init_states,
        jnp.zeros(n_envs, dtype=jnp.float32),
        jnp.zeros(n_envs, dtype=jnp.bool_),
        jnp.int32(0),
    )

    def _body(carry, _):
        keys, states, cum_rew, dones, step_idx = carry
        keys, sk = jax.vmap(jax.random.split)(keys).swapaxes(0, 1)
        n_actions = game.action_space.n
        actions = jax.vmap(
            lambda k: jax.random.randint(k, shape=(), minval=0, maxval=n_actions)
        )(sk)
        _, new_states, rewards, new_dones, _ = jax.vmap(game.step, in_axes=(0, 0, 0, None))(
            sk, states, actions, params
        )
        cum_rew = cum_rew + jnp.where(dones, jnp.float32(0.0), rewards)
        dones = dones | new_dones
        return (keys, new_states, cum_rew, dones, step_idx + jnp.int32(1)), None

    (_, _, returns, _, _), _ = jax.lax.scan(_body, init_carry, None, length=max_steps)
    return float(np.mean(np.asarray(returns)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("game_id,band", list(_BANDS.items()))
@pytest.mark.slow
def test_ale_fidelity(game_id: str, band: tuple[float, float]) -> None:
    """Mean return over N=1000 episodes must fall within the calibrated band."""
    lo, hi = band
    mean_return = _run_jax(game_id, _N_ENVS, _MAX_STEPS, _SEED)
    assert lo <= mean_return <= hi, (
        f"{game_id}: mean_return={mean_return:.2f} outside band [{lo}, {hi}]"
    )
