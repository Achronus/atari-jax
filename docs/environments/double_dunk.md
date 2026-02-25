# Double Dunk

> Game ID: `"atari/double_dunk-v0"`

2-on-2 half-court basketball. Score baskets for points; the CPU team defends and attacks. First to 24 points wins.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (shoot / steal) |
| `2` | UP (jump) |
| `3` | RIGHT |
| `4` | DOWN (move to position) |
| `5` | LEFT |

## Reward

| Event | Reward |
| --- | --- |
| 2-point basket | +2 |
| 3-point basket | +3 |

Shots taken from beyond x = 50 (the three-point line) award 3 points. All other successful shots award 2 points.

## Episode End

The episode ends when either the player or the CPU team reaches 24 points.

## Lives

No lives system. `lives` is always `0`.

## Screen Geometry

| Element | Position |
| --- | --- |
| Court | x ∈ [10, 150], y ∈ [30, 180] |
| Basket | x = 130, y = 70 |
| Three-point line | x = 50 |
| Ground level | y = 170 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/double_dunk-v0")
play("atari/double_dunk-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (shoot / steal) |
| `↑` / `W` | UP (jump) |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
