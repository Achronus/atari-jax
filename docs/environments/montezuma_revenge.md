# Montezuma's Revenge

> Game ID: `"atari/montezuma_revenge-v0"`

Panama Joe must navigate a complex Aztec pyramid collecting keys, avoiding
enemies, and solving puzzles to reach the treasure.  Climb ladders, jump
platforms, and collect all items to clear each room.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(8)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — jump |
| `2` | UP — climb ladder up |
| `3` | RIGHT |
| `4` | DOWN — climb ladder down |
| `5` | LEFT |
| `6` | UP + RIGHT |
| `7` | UP + LEFT |

## Reward

Points are awarded for collecting keys, items, and clearing rooms.

| Event | Points |
| --- | --- |
| Key collected | +100 |
| Item collected | +200 |
| Room cleared (all items) | +300 |

## Episode End

The episode ends when all lives are lost.  A life is lost when an enemy
touches the player.

## Lives

The player starts with 5 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Ground | y = 175 |
| Platforms | y = 80, 130, 175 |
| Ladders | x = 40, 80, 120 (y ∈ [80, 175]) |
| Player start | x = 20, y = ground level |
| Enemies (3) | patrol at ground level |
| Items (6) | 3 on upper platform (y = 70), 3 on mid platform (y = 125) |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/montezuma_revenge-v0")
play("atari/montezuma_revenge-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Jump |
| `↑` / `W` | Climb up |
| `→` / `D` | Move right |
| `↓` / `S` | Climb down |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
