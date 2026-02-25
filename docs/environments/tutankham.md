# Tutankham

> Game ID: `"atari/tutankham-v0"`

Navigate a multi-room Egyptian tomb, collect treasures, and shoot enemies. Enemies are birds and snakes; a limited-use magic wand can clear all enemies on screen. Collect all treasures in a room to advance.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(9)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (shoot laser — only fires left/right) |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |
| `6` | MAGIC WAND (kills all enemies on screen) |
| `7` | FIRE+RIGHT |
| `8` | FIRE+LEFT |

## Reward

| Event | Reward |
| --- | --- |
| Enemy shot | `+100` |
| Treasure collected | `+50` |
| Room clear bonus | `+200` |

## Episode End

The episode ends when all lives are lost. A life is lost by being touched by an enemy.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Room left | x = 10 |
| Room width | 140 px |
| Room top | y = 30 |
| Room height | 150 px |
| Player spawn | x = 20, y = 100 |
| Magic wand uses | 3 (per reset) |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/tutankham-v0")
play("atari/tutankham-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (shoot laser) |
| `↑` / `W` | UP |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN |
| `←` / `A` | LEFT |
| `Left Shift` | MAGIC WAND |
| `Esc` / close window | Quit |
