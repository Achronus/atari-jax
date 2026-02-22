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

"""
Single-environment throughput benchmark.

Measures step FPS for a single JIT-compiled `AtariEnv` by running a tight
Python step loop and blocking until all XLA computations are complete.

Usage
-----
    uv run benchmarks/bench_single_env.py
    uv run benchmarks/bench_single_env.py --game pong --steps 2000
"""

import argparse
import time

import jax
import jax.numpy as jnp

from atarax.env import make


def run(game: str, n_steps: int, n_warmup: int) -> None:
    """
    Run the single-env benchmark and print results.

    Parameters
    ----------
    game : str
        ALE game name (e.g. `"breakout"`).
    n_steps : int
        Number of timed steps to execute.
    n_warmup : int
        Number of warmup steps to run before timing starts.
    """
    env = make(f"atari/{game}-v0", preset=True, jit_compile=True)
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key)
    action = env.sample(key)

    # --- warmup (absorbs JIT compilation) ----------------------------------
    for _ in range(n_warmup):
        obs, state, _, _, _ = env.step(state, action)
    jax.block_until_ready(obs)

    # --- timed run ---------------------------------------------------------
    t0 = time.perf_counter()
    for _ in range(n_steps):
        obs, state, _, _, _ = env.step(state, action)
    jax.block_until_ready(obs)
    elapsed = time.perf_counter() - t0

    fps = n_steps / elapsed

    print(f"game        : {game}")
    print(f"obs shape   : {obs.shape}")
    print(f"n_steps     : {n_steps:,}")
    print(f"warmup      : {n_warmup}")
    print("\u2500" * 34)
    print(f"elapsed     : {elapsed:.3f} s")
    print(f"FPS         : {fps:,.0f} fps")
    print(f"target      : > 50,000 fps (GPU)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-env throughput benchmark.")
    parser.add_argument("--game", default="breakout", help="ALE game name")
    parser.add_argument("--steps", type=int, default=1000, help="Timed steps")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    args = parser.parse_args()

    run(args.game, args.steps, args.warmup)
