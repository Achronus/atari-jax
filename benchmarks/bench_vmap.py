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
Vmapped multi-environment throughput benchmark.

Measures step FPS for N parallel environments using `VecEnv.rollout()`, which
compiles a `lax.scan` loop over a vmapped step function.

Usage
-----
    uv run benchmarks/bench_vmap.py
    uv run benchmarks/bench_vmap.py --n 512 --steps 1000 --game pong
    uv run benchmarks/bench_vmap.py --n 64 --steps 200 --warmup 5
"""

import argparse
import time

import jax
import jax.numpy as jnp

from atarax.env import make_vec
from atarax.env._compile import DEFAULT_CACHE_DIR


def run(game: str, n_envs: int, n_steps: int, n_warmup: int) -> None:
    """
    Run the vmap benchmark and print results.

    Parameters
    ----------
    game : str
        ALE game name (e.g. `"breakout"`).
    n_envs : int
        Number of parallel environments.
    n_steps : int
        Number of steps per rollout (scan length).
    n_warmup : int
        Number of full rollouts to run before timing starts.
    """
    _backend = jax.default_backend()
    _cache = DEFAULT_CACHE_DIR / _backend
    print(f"device : {_backend}")
    print(f"cache  : {'warm' if _cache.is_dir() and any(_cache.iterdir()) else 'cold'}")

    vec_env = make_vec(
        f"atari/{game}-v0",
        n_envs=n_envs,
        preset=True,
        jit_compile=True,
        show_compile_progress=True,
    )
    key = jax.random.PRNGKey(0)

    _, states = vec_env.reset(key)
    actions = jnp.zeros((n_envs, n_steps), dtype=jnp.int32)

    # --- warmup (absorbs JIT compilation) ----------------------------------
    for _ in range(n_warmup):
        _, (obs, _, _, _) = vec_env.rollout(states, actions)
        jax.block_until_ready(obs)

    # --- timed run ---------------------------------------------------------
    t0 = time.perf_counter()
    _, (obs, _, _, _) = vec_env.rollout(states, actions)
    jax.block_until_ready(obs)
    elapsed = time.perf_counter() - t0

    total_frames = n_envs * n_steps
    fps = total_frames / elapsed
    fps_m = fps / 1_000_000

    print(f"game        : {game}")
    print(f"obs shape   : {obs.shape}")
    print(f"n_envs      : {n_envs:,}")
    print(f"n_steps     : {n_steps:,}")
    print(f"warmup runs : {n_warmup}")
    print("\u2500" * 34)
    print(f"total frames: {total_frames:,}")
    print(f"elapsed     : {elapsed:.3f} s")
    print(f"FPS         : {fps:,.0f} fps  ({fps_m:.2f} M)")
    print("target      : > 5,000,000 fps (512 envs, GPU)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vmapped multi-env throughput benchmark."
    )
    parser.add_argument("--game", default="breakout", help="ALE game name")
    parser.add_argument("--n", type=int, default=512, help="Number of parallel envs")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per rollout")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup rollouts")
    args = parser.parse_args()

    run(args.game, args.n, args.steps, args.warmup)
