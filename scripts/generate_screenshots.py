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
"""Generate per-game screenshots and an all_envs grid.

Steps each game ``--warmup`` frames before capturing so that motion trails
are visible in the output image.  Saves:

- ``assets/screenshots/<game>.png``   — uint8[210, 160, 3] native resolution
- ``assets/screenshots/all_envs.png`` — grid tiled with ``--nrow`` columns

Usage::

    uv run python scripts/generate_screenshots.py            # all games
    uv run python scripts/generate_screenshots.py --game breakout
    uv run python scripts/generate_screenshots.py --warmup 16 --seed 1
    uv run python scripts/generate_screenshots.py --out /tmp/shots --nrow 5
"""

import argparse
import pathlib

import imageio
import jax
import numpy as np

from atarax.env.registry import GAMES, PARAMS
from atarax.render import render_grid


def capture(game_id: str, seed: int, n_warmup: int) -> np.ndarray:
    """Reset a game, step ``n_warmup`` random non-NOOP actions, render.

    Returns
    -------
    frame : np.ndarray
        ``uint8[210, 160, 3]``
    """
    params = PARAMS[game_id]()
    game = GAMES[game_id]()
    key = jax.random.PRNGKey(seed)
    key, rk = jax.random.split(key)
    _, state = game.reset(rk, params)
    n_actions = game.action_space.n
    for _ in range(n_warmup):
        key, sk, ak = jax.random.split(key, 3)
        action = jax.random.randint(ak, shape=(), minval=1, maxval=n_actions)
        _, state, *_ = game.step(sk, state, action, params)
    return np.asarray(game.render(state))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate per-game screenshots and all_envs grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--game",
        default=None,
        metavar="GAME",
        help="Single game to capture (snake_case, e.g. breakout). Default: all.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for reset + warmup actions. Default: 0.",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="Number of random steps before capture (makes trails visible). Default: 8.",
    )
    ap.add_argument(
        "--nrow",
        type=int,
        default=4,
        help="Columns in the all_envs grid. Default: 4.",
    )
    ap.add_argument(
        "--out",
        default="assets/screenshots",
        metavar="DIR",
        help="Output directory. Default: assets/screenshots.",
    )
    ap.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip writing all_envs.png (useful when capturing a single game).",
    )
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    game_ids = [args.game] if args.game else sorted(GAMES.keys())
    frames = []

    print(f"Saving screenshots to {out}/  (warmup={args.warmup}, seed={args.seed})")
    for gid in game_ids:
        try:
            frame = capture(gid, args.seed, args.warmup)
            frames.append(frame)
            imageio.imwrite(out / f"{gid}.png", frame)
            print(f"  {gid}: {frame.shape}")
        except Exception as exc:
            print(f"  {gid}: skipped — {exc}")

    if frames and not args.no_grid:
        grid = render_grid(np.stack(frames), nrow=args.nrow)
        imageio.imwrite(out / "all_envs.png", grid)
        print(f"\nall_envs.png: {grid.shape}")

    print(f"\nDone. {len(frames)} screenshot(s) written.")


if __name__ == "__main__":
    main()
