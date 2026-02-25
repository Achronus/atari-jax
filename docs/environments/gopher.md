# Gopher

> Game ID: `"atari/gopher-v0"`

Protect three carrots from a tunnelling gopher. The gopher emerges from a tunnel on the left or right side, then moves toward a carrot and attempts to steal it. Shoot the gopher before it reaches the carrots.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (shoot) |
| `2` | UP (dig filler / move up) |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |

## Reward

| Event | Reward |
| --- | --- |
| Gopher shot | +200 |

## Episode End

The episode ends when all 3 carrots have been stolen by the gopher (lives reaches 0).

## Lives

`lives` tracks the number of surviving carrots (3 at start, decreasing as carrots are stolen). The episode ends when `lives` reaches 0.

## Screen Geometry

| Element | Position |
| --- | --- |
| Ground surface | y = 150 |
| Carrot y | y = 160 |
| Carrot positions (centre x) | x = 40, 80, 120 |
| Player y | y = 130 |
| Player x range | x ∈ [8, 144] |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/gopher-v0")
play("atari/gopher-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (shoot) |
| `→` / `D` | RIGHT |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
