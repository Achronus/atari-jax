# Venture

> Game ID: `"atari/venture-v0"`

Explore a multi-room dungeon, collecting treasures while fighting monsters. A large Hallmonster pursues you in the corridor; smaller monsters guard each room's treasure.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (shoot arrow) |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |

## Reward

| Event | Reward |
| --- | --- |
| Monster shot | `+100` |
| Treasure collected | `+100` |

## Episode End

The episode ends when all lives are lost. A life is lost by being touched by a room monster (when inside a room) or by the Hallmonster (when in the corridor).

## Lives

The player starts with 4 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Main corridor (hall) | y ∈ [90, 120] |
| Room area | y ∈ [30, 90], x ∈ [20, 140] |
| Room entrance x | x = 80 |
| Player x range | x ∈ [5, 150] |
| Player y range | y ∈ [30, 180] |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/venture-v0")
play("atari/venture-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (shoot arrow) |
| `↑` / `W` | UP |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
