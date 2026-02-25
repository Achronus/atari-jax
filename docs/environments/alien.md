# Alien

> Game ID: `"atari/alien-v0"`

Navigate a space station armed with a flamethrower, destroying alien eggs
scattered around the corridors while avoiding xenomorphs that chase you
through the rooms.  Shooting a xenomorph sends it back to its spawn point.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — shoot flamethrower |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |

## Reward

Points are awarded for each egg destroyed and for each alien shot.

| Event | Points |
| --- | --- |
| Egg destroyed | +10 |
| Alien shot | +30 |

## Episode End

The episode ends when all lives are lost.  A life is lost each time a
xenomorph catches the player.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Maze area | x ∈ [14, 146), y ∈ [21, 189) — 11 × 14 tiles of 12 × 12 px |
| Player start | centre of the maze |
| Egg locations | 6 fixed positions across the maze |
| Alien spawns | 2 aliens, each entering from opposite ends |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/alien-v0")
play("atari/alien-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire flamethrower |
| `↑` / `W` | Move up |
| `→` / `D` | Move right |
| `↓` / `S` | Move down |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
