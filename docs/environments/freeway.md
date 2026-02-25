# Freeway

> Game ID: `"atari/freeway-v0"`

Guide a chicken across 10 lanes of traffic to the opposite side of the screen. Each successful crossing earns +1. Cars push the chicken back on contact rather than costing a life. The episode ends after a fixed time limit.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(3)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | UP (move toward goal) |
| `2` | DOWN (move back toward start) |

## Reward

Each successful crossing of all 10 traffic lanes earns +1. Colliding with a car pushes the chicken back one lane; no reward penalty is applied.

## Episode End

The episode ends after 1600 emulated frames (400 agent steps). There is no lives-based termination.

## Lives

No lives system. `lives` is always `0`.

## Screen Geometry

| Element | Position |
| --- | --- |
| Road (10 lanes) | y ∈ [15, 195] |
| Lane height | 18 px |
| Chicken start | y = 192 |
| Goal | y <= 15 |
| Chicken x | x = 76 (fixed) |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/freeway-v0")
play("atari/freeway-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `↑` / `W` | UP (move toward goal) |
| `↓` / `S` | DOWN (move back) |
| `Esc` / close window | Quit |
