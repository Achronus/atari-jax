# Breakout

> Game ID: `"atari/breakout-v0"`

Use a paddle to bounce a ball upward and destroy all the coloured bricks arranged
in rows at the top of the screen.  Clear all bricks to advance to the next level
(bricks reset, lives and score carry over).

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(4)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — serve the ball |
| `2` | RIGHT — move paddle right |
| `3` | LEFT — move paddle left |

## Reward

Points are awarded for each brick destroyed.  The reward on each `step` call is
the sum earned across all four emulated frames in the frame-skip window.

| Brick row (top → bottom) | Points |
| --- | --- |
| Rows 0–1 (red) | 7 |
| Rows 2–3 (orange / yellow) | 4 |
| Rows 4–5 (green / blue) | 1 |

## Episode End

The episode ends when all five lives are lost.  A life is lost each time the ball
passes the paddle and falls off the bottom of the screen.

## Lives

The player starts with 5 lives.  The ball must be served with FIRE after each
life loss.

## Screen Geometry

| Element | Position |
| --- | --- |
| Playfield | x ∈ [8, 152), y ∈ [19, 210) |
| Brick area | y ∈ [57, 93), x ∈ [8, 152) — 6 rows × 18 columns × 8 × 6 px each |
| Paddle | y = 189, 16 × 4 px |
| Ball | 2 × 2 px |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/breakout-v0")               # 480×630 window, ALE-accurate speed
play("atari/breakout-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `→` / `D` | Move paddle right |
| `←` / `A` | Move paddle left |
| `Space` | Fire (serve ball) |
| `Esc` / close window | Quit |
