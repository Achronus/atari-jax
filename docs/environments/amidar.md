# Amidar

> Game ID: `"atari/amidar-v0"`

Paint every node of a grid by walking along paths while enemies patrol the
same routes.  The episode ends when all nodes are painted or all lives are
lost.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(5)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | UP |
| `2` | RIGHT |
| `3` | DOWN |
| `4` | LEFT |

## Reward

Points are awarded for painting new grid nodes.

| Event | Points |
| --- | --- |
| New node painted | +1 |

## Episode End

The episode ends when all lives are lost or all grid nodes have been
painted.  A life is lost each time an enemy catches the player.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Grid | 6 rows × 6 columns; nodes at x ∈ [10, 150], y ∈ [20, 160] |
| Player start | top-left node |
| Enemies | 3 enemies patrolling the grid paths |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/amidar-v0")
play("atari/amidar-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `↑` / `W` | Move up |
| `→` / `D` | Move right |
| `↓` / `S` | Move down |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
