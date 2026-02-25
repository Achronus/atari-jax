# Berzerk

> Game ID: `"atari/berzerk-v0"`

Navigate a maze of electrified walls shooting robots while evading the
invincible Evil Otto.  Touch a wall and you die; robots shoot back; Evil
Otto cannot be destroyed.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(9)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |
| `6` | UP + FIRE |
| `7` | RIGHT + FIRE |
| `8` | DOWN + FIRE |

## Reward

Points are awarded for each robot shot.

| Event | Points |
| --- | --- |
| Robot shot | +50 |

## Episode End

The episode ends when all lives are lost.  A life is lost by touching an
electrified wall, being shot by a robot, or being caught by Evil Otto.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Playfield | x ∈ [10, 150], y ∈ [30, 190] |
| Player start | x = 75, y = 105 |
| Robots | 6 robots arranged across the room |
| Evil Otto | enters from off-screen and closes on the player |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/berzerk-v0")
play("atari/berzerk-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `↑` / `W` | Move up |
| `→` / `D` | Move right |
| `↓` / `S` | Move down |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
