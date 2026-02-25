# Ms. Pac-Man

> Game ID: `"atari/ms_pacman-v0"`

Guide Ms. Pac-Man through a maze, eating dots and power pellets while avoiding
four coloured ghosts.  Eat a power pellet to frighten the ghosts and eat them
for bonus points.  Clear all dots and pellets to advance to the next level.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(5)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP (continue current direction) |
| `1` | UP |
| `2` | DOWN |
| `3` | LEFT |
| `4` | RIGHT |

## Reward

| Event | Reward |
| --- | --- |
| Eat a dot | `+10` |
| Eat a power pellet | `+50` |
| Eat a ghost (1st in chain) | `+200` |
| Eat a ghost (2nd in chain) | `+400` |
| Eat a ghost (3rd in chain) | `+800` |
| Eat a ghost (4th in chain) | `+1600` |
| Eat fruit (cherry) | `+100` |

## Episode End

The episode ends when all 3 lives are lost (`lives == 0`).

## Lives

The player starts with 3 lives.  A life is lost when a non-frightened ghost
occupies the same tile as Pac-Man.  After each death Pac-Man and all ghosts
return to their starting positions.

## Maze Layout

The maze is a fixed 28 × 31 tile grid (each tile = 5 × 5 pixels, centred in
the 160 × 210 frame).  Characters move one tile per agent step.

| Character | Start tile (col, row) |
| --- | --- |
| Pac-Man | (13, 23) |
| Blinky (red) | (13, 11) |
| Pinky (pink) | (14, 11) |
| Inky (cyan) | (13, 13) |
| Sue (orange) | (14, 13) |

Power pellets are located at `(row=23, col=4)` and `(row=23, col=23)`.

Fruit (cherry, +100) appears after 70 dots are eaten and despawns after
20 agent steps if not collected.

## Ghosts

All four ghosts chase Pac-Man by minimising L1 distance to his tile.  When
frightened (after a power pellet) they move randomly.  Frightened mode lasts
30 agent steps.  Reverse movement is penalised so ghosts prefer forward
momentum.

## Interactive Play

```python
from atarax.utils.render import play

play("atari/ms_pacman-v0")                     # 480×630 window, ALE-accurate speed
play("atari/ms_pacman-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `↑` / `W` | Move up |
| `↓` / `S` | Move down |
| `←` / `A` | Move left |
| `→` / `D` | Move right |
| `Esc` / close window | Quit |
