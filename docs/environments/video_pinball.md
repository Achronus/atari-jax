# Video Pinball

> Game ID: `"atari/video_pinball-v0"`

Classic pinball. Use left and right flippers to keep the ball in play and score points from bumpers, targets, and ramps.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (plunger — launch ball) |
| `2` | LEFT FLIPPER |
| `3` | RIGHT FLIPPER |
| `4` | LEFT FLIPPER + NUDGE |
| `5` | RIGHT FLIPPER + NUDGE |

## Reward

| Event | Reward |
| --- | --- |
| Bumper hit | `+100` |
| Target hit | `+500` |

When all 3 targets are hit, they reset along with all bumpers. Draining the ball (ball passes the bottom of the table) costs one ball but yields no reward.

## Episode End

The episode ends when all balls are lost (balls remaining reaches zero).

## Lives

The player starts with 3 balls. `lives` mirrors `balls_remaining` and decrements by 1 each time the ball drains past the flippers.

## Screen Geometry

| Element | Position |
| --- | --- |
| Table left wall | x = 15 |
| Table right wall | x = 145 |
| Table top | y = 20 |
| Table bottom | y = 195 |
| Left flipper | x ∈ [40, 65], y = 185 |
| Right flipper | x ∈ [95, 120], y = 185 |
| Bumpers (4) | (50, 80), (80, 60), (110, 80), (80, 110) |
| Targets (3) | (30, 40), (80, 40), (130, 40) |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/video_pinball-v0")
play("atari/video_pinball-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (launch ball) |
| `←` / `A` | LEFT FLIPPER |
| `→` / `D` | RIGHT FLIPPER |
| `Esc` / close window | Quit |
