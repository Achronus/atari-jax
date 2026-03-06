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
"""Re-calibrate ALE fidelity bands for reward and episode length.

Runs N random-policy episodes on both ALE (sequential) and atarax (vmapped)
and prints:

  1. A side-by-side comparison table (ALE vs JAX mean/std for return and steps)
  2. Ready-to-paste ``pytest.param`` blocks for ``test_ale_fidelity.py``

Band formula for both return and steps:  mean ± 3·SE  where SE = std / √N.
This gives ≈ 99.7% statistical coverage (CLT).

Steps bands are calibrated from the JAX run (same as return bands) and are
regression guards — the ALE step counts are shown alongside for comparison so
you can flag large dynamics mismatches before committing the bands.

Usage::

    uv run python scripts/calibrate_fidelity.py            # all registered games
    uv run python scripts/calibrate_fidelity.py --games breakout space_invaders
    uv run python scripts/calibrate_fidelity.py --jax-only --games asteroids

    # JAX-only mode skips ALE and compares against stored baselines:
    uv run python scripts/calibrate_fidelity.py --jax-only
    uv run python scripts/calibrate_fidelity.py --jax-only --games ms_pacman space_invaders

Requires ALE test dependencies (full mode only)::

    uv sync --group test
"""

import argparse
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

from atarax.env.registry import GAMES, PARAMS

# ---------------------------------------------------------------------------
# Stored ALE baselines (N=10 episodes, SEED=42, max_steps=1000, frame-skip=4)
# Used by --jax-only mode to avoid importing ale_py.
# ---------------------------------------------------------------------------
_ALE_BASELINES: dict[str, float] = {
    "asteroids": 754.5,
    "breakout": 1.1,
    "ms_pacman": 257.0,
    "space_invaders": 154.3,
}

# ---------------------------------------------------------------------------
# Constants — match the fidelity test exactly so bands are compatible
# ---------------------------------------------------------------------------
_N_ENVS = 1000
_N_ALE = 10       # ALE is sequential; 10 episodes is sufficient for a baseline check
_MAX_STEPS = 1_000
_SEED = 42

_ALL_GAMES = list(GAMES.keys())


# ---------------------------------------------------------------------------
# ALE random-policy runner (sequential, CPU)
# ---------------------------------------------------------------------------
def _run_ale(game_id: str, n_episodes: int, max_steps: int, seed: int):
    """Run ``n_episodes`` random-policy episodes in ALE.

    Returns
    -------
    returns : np.ndarray  shape (n_episodes,)
    steps   : np.ndarray  shape (n_episodes,)  agent steps (frame-skip=4)
    """
    ale = ale_py.ALEInterface()
    ale.setLoggerMode(ale_py.LoggerMode.Error)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("random_seed", seed)
    ale.loadROM(ale_py.roms.get_rom_path(game_id))

    legal = ale.getMinimalActionSet()
    rng = np.random.default_rng(seed)

    returns = np.zeros(n_episodes, dtype=np.float64)
    steps = np.zeros(n_episodes, dtype=np.int64)

    for ep in range(n_episodes):
        ale.reset_game()
        total_reward = 0.0
        ep_steps = 0

        while not ale.game_over() and ep_steps < max_steps:
            action = rng.choice(legal)
            # Frame-skip=4 to match atarax
            for _ in range(4):
                total_reward += ale.act(action)
                if ale.game_over():
                    break
            ep_steps += 1

        returns[ep] = total_reward
        steps[ep] = ep_steps if ale.game_over() else max_steps

    return returns, steps


# ---------------------------------------------------------------------------
# atarax random-policy runner (vmapped JAX)
# ---------------------------------------------------------------------------
def _run_jax(game_id: str, n_envs: int, max_steps: int, seed: int):
    """Run ``n_envs`` random-policy episodes in atarax via vmap + lax.scan.

    Returns
    -------
    returns    : np.ndarray  shape (n_envs,)
    steps      : np.ndarray  shape (n_envs,)
    iqm_return : float
    """
    game_cls = GAMES[game_id]
    game = game_cls()
    params = PARAMS[game_id]()
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)

    _, init_states = jax.vmap(game.reset, in_axes=(0, None))(keys, params)

    init_carry = (
        keys,
        init_states,
        jnp.zeros(n_envs, dtype=jnp.float32),
        jnp.zeros(n_envs, dtype=jnp.bool_),
        jnp.full(n_envs, max_steps, dtype=jnp.int32),
        jnp.int32(0),
    )

    def _body(carry, _):
        keys, states, cum_rew, dones, step_counts, step_idx = carry

        splits = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
        act_keys = splits[:, 0]
        env_keys = splits[:, 1]

        actions = jax.vmap(
            lambda k: jax.random.randint(k, shape=(), minval=0, maxval=game.num_actions)
        )(act_keys)

        _, new_states, rewards, new_dones, _ = jax.vmap(
            game.step, in_axes=(0, 0, 0, None)
        )(env_keys, states, actions, params)

        cum_rew = cum_rew + jnp.where(dones, jnp.float32(0.0), rewards)

        first_done = (~dones) & new_dones
        step_counts = jnp.where(first_done, step_idx + 1, step_counts)
        dones = dones | new_dones

        new_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1))(env_keys)
        return (new_keys, new_states, cum_rew, dones, step_counts, step_idx + 1), None

    (_, _, final_rew, _, final_steps, _), _ = jax.lax.scan(
        _body, init_carry, None, length=max_steps
    )

    final_rew_np = np.asarray(final_rew)
    final_steps_np = np.asarray(final_steps)

    sorted_r = np.sort(final_rew_np)
    lo, hi = n_envs // 4, 3 * n_envs // 4
    iqm = float(np.mean(sorted_r[lo:hi]))

    return final_rew_np, final_steps_np, iqm


# ---------------------------------------------------------------------------
# Band helpers
# ---------------------------------------------------------------------------
def _band(mean: float, std: float, n: int, sigma: float = 3.0):
    """Return (lo, hi) = mean ± sigma·SE where SE = std/√n."""
    se = std / math.sqrt(n)
    return mean - sigma * se, mean + sigma * se


def _fmt_band(lo: float, hi: float, decimals: int = 1) -> str:
    fmt = f".{decimals}f"
    return f"[{lo:{fmt}}, {hi:{fmt}}]"


# ---------------------------------------------------------------------------
# Per-game calibration
# ---------------------------------------------------------------------------
def calibrate_game(
    game_id: str,
    n: int,
    max_steps: int,
    seed: int,
    n_ale: int = 10,
    jax_only: bool = False,
) -> dict:
    if jax_only:
        ale_mean_r = _ALE_BASELINES.get(game_id, float("nan"))
        ale_std_r = float("nan")
        ale_mean_s = float("nan")
        ale_std_s = float("nan")
        print(f"  {game_id}: JAX...", end="", flush=True)
    else:
        print(f"  {game_id}: ALE...", end="", flush=True)
        ale_ret, ale_steps = _run_ale(game_id, n_ale, max_steps, seed)
        ale_mean_r, ale_std_r = float(np.mean(ale_ret)), float(np.std(ale_ret))
        ale_mean_s, ale_std_s = float(np.mean(ale_steps)), float(np.std(ale_steps))
        print(" JAX...", end="", flush=True)

    jax_ret, jax_steps, jax_iqm = _run_jax(game_id, n, max_steps, seed)
    print(" done.")

    jax_mean_r, jax_std_r = float(np.mean(jax_ret)), float(np.std(jax_ret))
    jax_mean_s, jax_std_s = float(np.mean(jax_steps)), float(np.std(jax_steps))

    ret_lo, ret_hi = _band(jax_mean_r, jax_std_r, n)
    steps_lo, steps_hi = _band(jax_mean_s, jax_std_s, n)

    return {
        "game_id": game_id,
        "ale_mean_r": ale_mean_r,
        "ale_std_r": ale_std_r,
        "ale_mean_s": ale_mean_s,
        "ale_std_s": ale_std_s,
        "jax_mean_r": jax_mean_r,
        "jax_std_r": jax_std_r,
        "jax_iqm": jax_iqm,
        "jax_mean_s": jax_mean_s,
        "jax_std_s": jax_std_s,
        "ret_lo": ret_lo,
        "ret_hi": ret_hi,
        "steps_lo": steps_lo,
        "steps_hi": steps_hi,
    }


# ---------------------------------------------------------------------------
# Report formatters
# ---------------------------------------------------------------------------
def _print_table(results: list[dict]) -> None:
    header = (
        f"{'Game':<16}  {'ALE ret':>9}  {'JAX ret':>9}  {'JAX IQM':>9}  "
        f"{'ALE steps':>9}  {'JAX steps':>9}  {'ret band':>18}  {'steps band':>22}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        rb = _fmt_band(r["ret_lo"], r["ret_hi"])
        sb = _fmt_band(r["steps_lo"], r["steps_hi"], decimals=0)
        print(
            f"{r['game_id']:<16}  "
            f"{r['ale_mean_r']:>9.2f}  "
            f"{r['jax_mean_r']:>9.2f}  "
            f"{r['jax_iqm']:>9.2f}  "
            f"{r['ale_mean_s']:>9.1f}  "
            f"{r['jax_mean_s']:>9.1f}  "
            f"{rb:>18}  "
            f"{sb:>22}"
        )
    print(sep + "\n")


def _print_pytest_params(results: list[dict]) -> None:
    print("# ---- paste into test_ale_fidelity.py @pytest.mark.parametrize ----")
    for r in results:
        gid = r["game_id"]
        ret_lo = round(r["ret_lo"], 1)
        ret_hi = round(r["ret_hi"], 1)
        steps_lo = max(0, int(math.floor(r["steps_lo"])))
        steps_hi = int(math.ceil(r["steps_hi"]))
        print(
            f"        # {gid}: ALE ret={r['ale_mean_r']:.2f} | "
            f"JAX ret={r['jax_mean_r']:.2f} ± {r['jax_std_r']:.2f}  "
            f"ALE steps={r['ale_mean_s']:.0f} | "
            f"JAX steps={r['jax_mean_s']:.0f} ± {r['jax_std_s']:.0f}"
        )
        print(
            f"        pytest.param(\"{gid}\", {ret_lo}, {ret_hi}, "
            f"{steps_lo}, {steps_hi}, id=\"{gid}\"),"
        )
    print("# -------------------------------------------------------------------")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate ALE fidelity bands (return + episode steps).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--games",
        nargs="+",
        default=None,
        metavar="GAME",
        help="Games to calibrate (space-separated snake_case). Default: all.",
    )
    ap.add_argument("--n", type=int, default=_N_ENVS, help=f"JAX episodes per game (vmapped). Default: {_N_ENVS}.")
    ap.add_argument("--n-ale", type=int, default=_N_ALE, help=f"ALE episodes per game (sequential). Default: {_N_ALE}.")
    ap.add_argument("--max-steps", type=int, default=_MAX_STEPS, help=f"Max steps per episode. Default: {_MAX_STEPS}.")
    ap.add_argument("--seed", type=int, default=_SEED, help=f"PRNG seed. Default: {_SEED}.")
    ap.add_argument(
        "--jax-only",
        action="store_true",
        help="Skip ALE run; compare JAX results against stored baselines in _ALE_BASELINES. Faster — no ale_py required.",
    )
    args = ap.parse_args()

    if not args.jax_only:
        import ale_py  # noqa: F401 — validate ale_py is available before starting

    games = args.games if args.games else _ALL_GAMES
    invalid = [g for g in games if g not in GAMES]
    if invalid:
        print(f"Unknown games: {invalid}. Valid: {_ALL_GAMES}", file=sys.stderr)
        sys.exit(1)

    mode = "JAX-only (stored ALE baselines)" if args.jax_only else f"JAX + ALE (N={args.n_ale})"
    print(f"Calibrating {len(games)} game(s) — N={args.n} (JAX), seed={args.seed}, max_steps={args.max_steps} [{mode}]")
    results = []
    for gid in games:
        try:
            results.append(
                calibrate_game(gid, args.n, args.max_steps, args.seed, n_ale=args.n_ale, jax_only=args.jax_only)
            )
        except Exception as exc:
            print(f"  {gid}: FAILED — {exc}")

    if not results:
        print("No results to report.")
        return

    _print_table(results)
    _print_pytest_params(results)


if __name__ == "__main__":
    main()
