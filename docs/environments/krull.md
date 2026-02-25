# Krull

> Game ID: `"atari/krull-v0"`

Prince Colwyn must navigate a series of challenges to rescue Princess
Lyssa, using the magical Glaive as a weapon.  Throw the Glaive at enemies;
it bounces off walls and returns to the player.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(10)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — throw Glaive |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |
| `6` | UP + RIGHT |
| `7` | UP + LEFT |
| `8` | DOWN + RIGHT |
| `9` | DOWN + LEFT |

## Reward

Points are awarded for each enemy hit by the Glaive.

| Event | Points |
| --- | --- |
| Enemy hit | +50 |

## Episode End

The episode ends when all lives are lost.  A life is lost when an enemy
reaches the player.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Arena | x ∈ [5, 155], y ∈ [20, 195] |
| Player start | x = 80, y = 140 |
| Enemies (6) | start spread across rows y = 40–80 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/krull-v0")
play("atari/krull-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Throw Glaive |
| `↑` / `W` | Move up |
| `→` / `D` | Move right |
| `↓` / `S` | Move down |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
