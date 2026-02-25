# Skiing

> Game ID: `"atari/skiing-v0"`

Slalom ski down a mountain, passing through gates as fast as possible. Miss a gate and receive a time penalty. Lowest elapsed time wins.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(5)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | RIGHT (lean right) |
| `2` | LEFT (lean left) |
| `3` | HARD RIGHT (sharp turn) |
| `4` | HARD LEFT (sharp turn) |

## Reward

The reward is `-1` per frame. Each missed gate adds a further `-300` penalty on the frame it is missed. There are 20 gates in total. The episode reward therefore encodes elapsed time plus penalties, so a higher (less negative) cumulative reward corresponds to a faster, cleaner run.

## Episode End

The episode ends when all 20 gates have been either passed or missed.

## Lives

No lives system. `lives` is always `0`.

## Screen Geometry

| Element | Position |
| --- | --- |
| Player (skier) | fixed at y = 50; mountain scrolls beneath |
| Gate opening width | 20 px |
| Gate post width | 4 px |
| Player x range | x ∈ [5, 155] |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/skiing-v0")
play("atari/skiing-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `→` / `D` | RIGHT (lean right) |
| `←` / `A` | LEFT (lean left) |
| `Esc` / close window | Quit |
