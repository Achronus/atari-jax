# Pong

> Game ID: `"atari/pong-v0"`

Volley a ball back and forth against a CPU opponent using a paddle on the right
side of the court.  The first player to reach 21 points wins the match.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(3)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | UP — move paddle up |
| `2` | DOWN — move paddle down |

## Reward

The reward on each `step` call reflects the change in score differential across
the four emulated frames in the frame-skip window.

| Event | Reward |
| --- | --- |
| Player scores a point | `+1` |
| CPU scores a point | `−1` |
| No scoring | `0` |

## Episode End

The episode ends when either player reaches 21 points (`score >= 21` or
`opp_score >= 21`).

## Lives

No lives system.  `lives` is always `0`.  The episode ends through the score
limit, not life loss.

## CPU Opponent

The CPU paddle tracks the ball centre at a fixed maximum speed of 1.5 pixels
per emulated frame (6 px per agent step at 4× frame-skip).  This makes the CPU
beatable but non-trivial: aiming for the edges of the CPU paddle causes
deflections the CPU cannot react to in time.

## Screen Geometry

| Element | Position |
| --- | --- |
| Court | y ∈ [34, 194), x ∈ [8, 152) |
| Player paddle (right, green) | x ∈ [148, 152), h=16 px |
| CPU paddle (left, red) | x ∈ [8, 12), h=16 px |
| Ball | 2 × 4 px |
| Net | x = 80, dashed every 4 px |

Both paddles can move in y ∈ [34, 178) (clamped so the paddle stays in-court).

## Interactive Play

```python
from atarax.utils.render import play

play("atari/pong-v0")                        # 480×630 window, ALE-accurate speed
play("atari/pong-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `↑` / `W` | Move paddle up |
| `↓` / `S` | Move paddle down |
| `Esc` / close window | Quit |
