# Robotank

> Game ID: `"atari/robotank-v0"`

Command a robot tank platoon to destroy enemy robot tanks.  The player
aims and fires from a top-down perspective.  Enemy tanks approach from all
sides; shoot them before they destroy your tanks.

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
| `2` | UP — aim up |
| `3` | RIGHT — aim right |
| `4` | DOWN — aim down |
| `5` | LEFT — aim left |
| `6` | UP + FIRE |
| `7` | RIGHT + FIRE |
| `8` | DOWN + FIRE |

## Reward

Points are awarded for each enemy tank destroyed.

| Event | Points |
| --- | --- |
| Enemy tank destroyed | +1 |

## Episode End

The episode ends when all lives (tanks) are lost.  A life is lost when an
enemy bullet hits the player tank or an enemy tank reaches the player.

## Lives

The player starts with 5 lives (tanks).

## Screen Geometry

| Element | Position |
| --- | --- |
| Arena | full screen |
| Player tank | x = 80, y = 105 (centre) |
| Enemy tanks (8) | start at corners and edges, close in on player |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/robotank-v0")
play("atari/robotank-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `↑` / `W` | Aim up |
| `→` / `D` | Aim right |
| `↓` / `S` | Aim down |
| `←` / `A` | Aim left |
| `Esc` / close window | Quit |
