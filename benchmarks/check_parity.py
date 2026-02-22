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
Parity check: JAX environment vs. ALE reference implementation.

Runs both environments for the same fixed action sequence and compares
rewards, terminal signals, and screen pixel statistics.  Exact frame-for-frame
pixel agreement is not expected (seeding and TIA rendering can differ slightly),
but rewards and done flags should match for the same game events, and screens
should have similar statistical properties.

Usage
-----
    uv run benchmarks/check_parity.py
    uv run benchmarks/check_parity.py --game pong --steps 500
    uv run benchmarks/check_parity.py --game breakout --steps 1000 --seed 7
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from ale_py import ALEInterface
from ale_py.roms import get_rom_path
from tqdm import tqdm

from atarax.env._compile import _live_bar, setup_cache
from atarax.env.atari_env import AtariEnv, EnvParams


def _run_jax(ale_name: str, actions: list[int], seed: int) -> dict:
    """Run the JAX environment for the given action sequence."""
    setup_cache()
    params = EnvParams(noop_max=0)
    env = AtariEnv(ale_name, params)
    key = jax.random.PRNGKey(seed)

    # First reset triggers XLA kernel compilation — show live progress.
    with tqdm(total=1, desc="Compiling JAX kernel", leave=True) as bar:
        with _live_bar(bar):
            _, state = env.reset(key)
        bar.update(1)

    rewards, dones, screens = [], [], []

    for a in actions:
        _, state, reward, done, _ = env.step(state, jnp.int32(a))
        rewards.append(float(reward))
        dones.append(bool(done))
        screens.append(np.asarray(state.screen))
        if done:
            _, state = env.reset(key)

    return {"rewards": rewards, "dones": dones, "screens": screens}


def _run_ale(ale_name: str, actions: list[int], seed: int) -> dict:
    """Run the ALE reference implementation for the given action sequence."""
    ale = ALEInterface()
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setBool("display_screen", False)
    ale.loadROM(get_rom_path(ale_name))

    rewards, dones, screens = [], [], []

    for a in actions:
        reward = ale.act(a)
        done = ale.game_over()
        screen = ale.getScreenRGB().copy()
        rewards.append(float(reward))
        dones.append(bool(done))
        screens.append(screen)

        if done:
            ale.reset_game()

    return {"rewards": rewards, "dones": dones, "screens": screens}


def _print_header(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _compare(jax_data: dict, ale_data: dict, n_steps: int) -> None:
    rewards_jax = np.array(jax_data["rewards"])
    rewards_ale = np.array(ale_data["rewards"])
    dones_jax = np.array(jax_data["dones"])
    dones_ale = np.array(ale_data["dones"])
    screens_jax = np.stack(jax_data["screens"])  # [T, 210, 160, 3]
    screens_ale = np.stack(ale_data["screens"])

    # --- Rewards ---
    _print_header("Rewards")
    print(
        f"  JAX  total : {rewards_jax.sum():.1f}  (non-zero steps: {(rewards_jax != 0).sum()})"
    )
    print(
        f"  ALE  total : {rewards_ale.sum():.1f}  (non-zero steps: {(rewards_ale != 0).sum()})"
    )
    reward_match = np.array_equal(rewards_jax, rewards_ale)
    print(f"  Exact match: {'YES' if reward_match else 'NO'}")
    if not reward_match:
        diff_steps = np.where(rewards_jax != rewards_ale)[0]
        print(f"  First 5 diverging steps: {diff_steps[:5].tolist()}")
        for i in diff_steps[:5]:
            print(
                f"    step {i:4d}  JAX={rewards_jax[i]:.1f}  ALE={rewards_ale[i]:.1f}"
            )

    # --- Done flags ---
    _print_header("Terminal signals (done)")
    done_match = np.array_equal(dones_jax, dones_ale)
    jax_done_steps = np.where(dones_jax)[0].tolist()
    ale_done_steps = np.where(dones_ale)[0].tolist()
    print(f"  JAX  done at steps: {jax_done_steps[:10]}")
    print(f"  ALE  done at steps: {ale_done_steps[:10]}")
    print(f"  Exact match: {'YES' if done_match else 'NO'}")

    # --- Screen pixel statistics ---
    _print_header("Screen pixel statistics")
    print(f"  {'':20s}  {'JAX':>12s}  {'ALE':>12s}")
    print(f"  {'mean':20s}  {screens_jax.mean():>12.2f}  {screens_ale.mean():>12.2f}")
    print(f"  {'std':20s}  {screens_jax.std():>12.2f}  {screens_ale.std():>12.2f}")
    print(f"  {'min':20s}  {screens_jax.min():>12d}  {screens_ale.min():>12d}")
    print(f"  {'max':20s}  {screens_jax.max():>12d}  {screens_ale.max():>12d}")
    print(
        f"  {'non-zero pixels (%)':20s}  "
        f"{100 * (screens_jax > 0).mean():>11.1f}%  "
        f"{100 * (screens_ale > 0).mean():>11.1f}%"
    )

    # Mean absolute error between corresponding frames
    mae = np.abs(screens_jax.astype(np.int32) - screens_ale.astype(np.int32)).mean()
    print(f"\n  Mean absolute pixel error (JAX vs ALE): {mae:.2f}")
    if mae < 5.0:
        print("  -> Screens are very close (MAE < 5).")
    elif mae < 20.0:
        print("  -> Screens differ slightly — likely seeding or TIA timing.")
    else:
        print("  -> Screens differ substantially — investigate TIA rendering.")

    _print_header("Summary")
    ok = reward_match and done_match
    print(f"  Rewards match  : {'PASS' if reward_match else 'FAIL'}")
    print(f"  Done flags match: {'PASS' if done_match else 'FAIL'}")
    print(f"  Overall        : {'PASS' if ok else 'FAIL (see above)'}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX vs ALE parity check")
    parser.add_argument("--game", default="breakout", help="ALE game name")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Fixed repeating action sequence: FIRE to start, then LEFT/RIGHT/NOOP
    pattern = [1, 4, 4, 3, 3, 0, 0]  # FIRE, LEFT, LEFT, RIGHT, RIGHT, NOOP, NOOP
    actions = (pattern * (args.steps // len(pattern) + 1))[: args.steps]

    print(f"Game    : {args.game}")
    print(f"Steps   : {args.steps}")
    print(f"Seed    : {args.seed}")
    print(f"Actions : pattern {pattern!r} repeated")

    print("\n[JAX] Running...")
    jax_data = _run_jax(args.game, actions, args.seed)
    print("[JAX] Done.")

    print("\n[ALE] Running...")
    ale_data = _run_ale(args.game, actions, args.seed)
    print("[ALE] Done.")

    _compare(jax_data, ale_data, args.steps)


if __name__ == "__main__":
    main()
