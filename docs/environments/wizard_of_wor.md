# Wizard of Wor

> Game ID: `"atari/wizard_of_wor-v0"`

Navigate a dungeon maze and destroy Worlings, Garwors, Thorwors, and ultimately the Wizard himself. Enemies can become invisible. This JAX version is single-player.

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
| `6` | FIRE+UP |
| `7` | FIRE+RIGHT |
| `8` | FIRE+LEFT |

## Reward

| Event | Reward |
| --- | --- |
| Worling destroyed | `+100` |
| Garwor destroyed | `+200` |
| Thorwor destroyed | `+500` |
| Wizard destroyed | `+2500` |

## Episode End

The episode ends when all lives are lost. A life is lost when the player is hit by an enemy bullet or touched by an enemy.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Dungeon left | x = 10 |
| Dungeon right | x = 150 |
| Dungeon top | y = 30 |
| Dungeon bottom | y = 185 |
| Player spawn | x = 80, y = 170 |
| Enemy types | 0 = Worling, 1 = Garwor, 2 = Thorwor, 3 = Thorwor, 4 = Wizard |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/wizard_of_wor-v0")
play("atari/wizard_of_wor-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE |
| `↑` / `W` | UP |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
