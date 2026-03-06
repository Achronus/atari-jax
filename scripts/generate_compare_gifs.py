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
"""Generate side-by-side comparison GIFs: atarax (left) vs ALE ROM (right).

Each game produces one file saved to ``assets/compare/``:

- ``<game>_compare.gif`` — animated comparison, 2× upscaled for readability

Both sides run an independent random policy for ``--steps`` agent steps.
The atarax side uses ``jax.random``; the ALE side uses ``numpy.random``.

Usage::

    uv run python scripts/generate_compare_gifs.py            # all 15 games
    uv run python scripts/generate_compare_gifs.py --game breakout
    uv run python scripts/generate_compare_gifs.py --steps 60 --fps 15

Requires ALE and viz dependencies::

    uv sync --group test
    pip install imageio
"""

import argparse
import pathlib

import ale_py
import imageio
import jax
import numpy as np
from PIL import Image

from atarax.env.registry import GAMES, PARAMS

_SEPARATOR_W = 6   # px wide grey bar between the two renders
_UPSCALE = 2       # integer upscale for readability in markdown

# Maze navigator games use 1 tile/step internally (no frame skip).
_MAZE_GAMES: frozenset[str] = frozenset({"ms_pacman"})


def _ale_frames(game_id: str, n_steps: int, seed: int, frame_skip: int = 4) -> list[np.ndarray]:
    """Run n_steps of random play in ALE, return one uint8[210,160,3] per step."""
    ale = ale_py.ALEInterface()
    ale.setLoggerMode(ale_py.LoggerMode.Error)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("random_seed", seed)
    ale.loadROM(ale_py.roms.get_rom_path(game_id))
    ale.reset_game()

    legal = ale.getMinimalActionSet()
    rng = np.random.default_rng(seed)
    frames = [ale.getScreenRGB().copy()]

    for _ in range(n_steps - 1):
        if ale.game_over():
            ale.reset_game()
        action = rng.choice(legal)
        for _ in range(frame_skip):          # match atarax frame skip (1 for maze games)
            ale.act(action)
            if ale.game_over():
                break
        frames.append(ale.getScreenRGB().copy())

    return frames   # list of uint8[210, 160, 3]


def _atarax_frames(game_id: str, n_steps: int, seed: int) -> list[np.ndarray]:
    """Run n_steps of random play in atarax, return one uint8[210,160,3] per step."""
    params = PARAMS[game_id]()
    game = GAMES[game_id]()
    key = jax.random.PRNGKey(seed)
    key, rk = jax.random.split(key)
    _, state = game.reset(rk, params)
    n_actions = game.action_space.n

    frames = [np.asarray(game.render(state))]

    for _ in range(n_steps - 1):
        key, sk, ak = jax.random.split(key, 3)
        action = jax.random.randint(ak, shape=(), minval=1, maxval=n_actions)
        _, state, _, done, _ = game.step(sk, state, action, params)
        if bool(done):
            key, rk = jax.random.split(key)
            _, state = game.reset(rk, params)
        frames.append(np.asarray(game.render(state)))

    return frames   # list of uint8[210, 160, 3]


def _composite(
    atarax_frame: np.ndarray,
    ale_frame: np.ndarray,
    upscale: int,
    sep_w: int,
) -> np.ndarray:
    """Combine two uint8[210,160,3] frames into a side-by-side uint8[H,W,3]."""
    sep = np.full((210, sep_w, 3), 100, dtype=np.uint8)
    panel = np.concatenate([atarax_frame, sep, ale_frame], axis=1)   # (210, 330, 3)
    if upscale > 1:
        h, w = panel.shape[:2]
        panel = np.asarray(
            Image.fromarray(panel).resize((w * upscale, h * upscale), Image.NEAREST)
        )
    return panel


def generate(game_id: str, n_steps: int, seed: int, out_dir: pathlib.Path) -> None:
    """Generate the comparison GIF for one game."""
    frame_skip = 1 if game_id in _MAZE_GAMES else 4
    atarax_fs = _atarax_frames(game_id, n_steps, seed)
    ale_fs = _ale_frames(game_id, n_steps, seed, frame_skip=frame_skip)

    # Zip (both have n_steps frames; if lengths differ, take the shorter)
    composites = [
        _composite(af, lf, _UPSCALE, _SEPARATOR_W)
        for af, lf in zip(atarax_fs, ale_fs)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{game_id}_compare.gif"
    imageio.mimsave(str(path), composites)
    h, w = composites[0].shape[:2]
    print(f"  {game_id}: {len(composites)} frames  {w}x{h}  -> {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate side-by-side atarax vs ALE comparison GIFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--game",
        default=None,
        metavar="GAME",
        help="Single game (snake_case). Default: all registered games.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=45,
        help="Agent steps per side. Default: 45 (~3 s at 15 fps).",
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=15,
        help="GIF frame rate. Default: 15.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed. Default: 0.",
    )
    ap.add_argument(
        "--out",
        default="assets/compare",
        metavar="DIR",
        help="Output directory. Default: assets/compare.",
    )
    args = ap.parse_args()

    game_ids = [args.game] if args.game else sorted(GAMES.keys())
    out = pathlib.Path(args.out)

    print(
        f"Generating comparison GIFs to {out}/  "
        f"(steps={args.steps}, fps={args.fps}, seed={args.seed})"
    )
    for gid in game_ids:
        try:
            generate(gid, args.steps, args.seed, out)
        except Exception as exc:
            print(f"  {gid}: skipped — {exc}")

    n = len(list(out.glob("*_compare.gif")))
    print(f"\nDone. {n} GIF(s) in {out}/")


if __name__ == "__main__":
    main()
