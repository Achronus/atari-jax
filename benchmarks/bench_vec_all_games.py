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
Multi-game VecEnv throughput benchmark.

Mirrors the meta-RL training pattern used in velora: an outer loop over N
passes with an inner loop stepping through all registered Atari games each
pass.  Each game runs `n_envs` parallel environments for `n_steps` steps via
`VecEnv.rollout()`.  States are carried across outer-loop passes so the
environments continue from where they left off.

Usage
-----
    uv run benchmarks/bench_vec_all_games.py
    uv run benchmarks/bench_vec_all_games.py --passes 20 --n-envs 8 --n-steps 64
    uv run benchmarks/bench_vec_all_games.py --verbose
"""

import argparse
import time
from typing import List

import jax
import jax.numpy as jnp
from tqdm import tqdm

from atarax import make_multi_vec
from atarax._compile import DEFAULT_CACHE_DIR
from atarax.env.registry import GAME_SPECS


def run(n_envs: int, n_steps: int, n_passes: int, verbose: bool) -> None:
    """
    Run the all-games benchmark and print a summary.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments per game (vmap width).
    n_steps : int
        Rollout length — steps collected per game per outer pass.
    n_passes : int
        Number of outer training-step iterations to time.
    verbose : bool
        Print per-game timing table after the run.
    """
    _backend = jax.default_backend()
    _cache = DEFAULT_CACHE_DIR / _backend
    n_games = len(GAME_SPECS)
    game_names = [spec.env_name for spec in GAME_SPECS]

    print(f"device : {_backend}")
    print(f"cache  : {'warm' if _cache.is_dir() and any(_cache.iterdir()) else 'cold'}")
    print(f"games  : {n_games}")

    # --- build + JIT-compile all VecEnvs ------------------------------------
    print(f"\nBuilding {n_games} VecEnvs (n_envs={n_envs}, jit_compile=True)...")
    envs = make_multi_vec(GAME_SPECS, n_envs=n_envs, jit_compile=True)

    # --- initial reset -------------------------------------------------------
    key = jax.random.PRNGKey(0)
    actions = jnp.zeros((n_envs, n_steps), dtype=jnp.int32)

    print("Resetting all environments...")
    all_states = [env.reset(key)[1] for env in envs]

    # --- warm-up (one extra rollout per env to prime XLA runtime caches) ----
    print("Warming up...")
    with tqdm(total=n_games, desc="Warmup", unit="env") as bar:
        for i, (env, state) in enumerate(zip(envs, all_states)):
            bar.set_description(f"Warmup: {game_names[i]}")
            new_state, _ = env.rollout(state, actions)
            all_states[i] = new_state
            bar.update(1)
    jax.block_until_ready(all_states[0])

    # --- timed outer loop ---------------------------------------------------
    print(
        f"\nTiming {n_passes} passes × {n_games} games × "
        f"{n_envs} envs × {n_steps} steps..."
    )

    pass_times: List[float] = []
    game_times: List[List[float]] = [[] for _ in range(n_games)]

    for _ in range(n_passes):
        t_pass_start = time.perf_counter()

        for i, (env, state) in enumerate(zip(envs, all_states)):
            t_game = time.perf_counter()
            new_state, _ = env.rollout(state, actions)
            jax.block_until_ready(new_state)
            all_states[i] = new_state
            game_times[i].append(time.perf_counter() - t_game)

        pass_times.append(time.perf_counter() - t_pass_start)

    # --- report -------------------------------------------------------------
    total_time = sum(pass_times)
    total_steps = n_games * n_envs * n_steps * n_passes
    steps_per_sec = total_steps / total_time
    ms_per_pass_mean = 1000 * total_time / n_passes
    ms_per_pass_min = 1000 * min(pass_times)
    ms_per_pass_max = 1000 * max(pass_times)

    sep = "\u2500" * 46
    print()
    print(sep)
    print(f"  n_games      : {n_games}")
    print(f"  n_envs       : {n_envs}")
    print(f"  n_steps      : {n_steps}")
    print(f"  n_passes     : {n_passes}")
    print(sep)
    print(f"  total steps  : {total_steps:,}")
    print(f"  total time   : {total_time:.3f} s")
    print(f"  steps/sec    : {steps_per_sec:,.0f}")
    print(
        f"  ms / pass    : {ms_per_pass_mean:.1f} mean  "
        f"[{ms_per_pass_min:.1f} \u2013 {ms_per_pass_max:.1f}]"
    )
    print(sep)

    if verbose:
        print()
        print(f"  {'Game':<24} {'Mean ms/pass':>12}  {'Min ms':>8}  {'Max ms':>8}")
        print("  " + "\u2500" * 57)
        for name, times in zip(game_names, game_times):
            mean_ms = 1000 * sum(times) / len(times)
            min_ms = 1000 * min(times)
            max_ms = 1000 * max(times)
            print(f"  {name:<24} {mean_ms:>12.2f}  {min_ms:>8.2f}  {max_ms:>8.2f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-game VecEnv throughput benchmark."
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        metavar="N",
        help="Parallel environments per game (default: 4)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=29,
        metavar="T",
        help="Rollout length per game per pass (default: 29)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=10,
        help="Number of outer loop passes to time (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-game timing table",
    )
    args = parser.parse_args()

    run(args.n_envs, args.n_steps, args.passes, args.verbose)
