# Enduro

> Game ID: `"atari/enduro-v0"`

Race a car through traffic across multiple days. Pass a required number of opponent cars each day to advance. Colliding with opponents slows you down.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(9)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (accelerate) |
| `2` | RIGHT |
| `3` | LEFT |
| `4` | DOWN (brake) |
| `5` | DOWN + RIGHT |
| `6` | DOWN + LEFT |
| `7` | RIGHT + FIRE |
| `8` | LEFT + FIRE |

## Reward

| Event | Reward |
| --- | --- |
| Opponent car passed | +1 |
| Day completed | +100 |

Each opponent car that scrolls past the player's position scores +1. Successfully passing 200 cars before the day timer expires awards +100 and advances to the next day.

## Episode End

The episode ends when the day timer runs out and the player has not passed the required 200 cars for that day.

## Lives

No lives system. `lives` is always `0`.

## Screen Geometry

| Element | Position |
| --- | --- |
| Road | x ∈ [20, 140] |
| Player car | y = 150, x ∈ [30, 130] |
| Opponent spawn y | y = 30 |
| Opponent despawn y | y = 190 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/enduro-v0")
play("atari/enduro-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (accelerate) |
| `→` / `D` | RIGHT |
| `←` / `A` | LEFT |
| `↓` / `S` | DOWN (brake) |
| `Esc` / close window | Quit |
