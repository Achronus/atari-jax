# Demon Attack

> Game ID: `"atari/demon_attack-v0"`

Destroy waves of demons descending from the sky using a cannon on an ice planet. Demons fire back; getting hit costs a life. Waves grow more difficult as enemies split into two when their row is shot out.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | RIGHT |
| `3` | LEFT |
| `4` | RIGHT + FIRE |
| `5` | LEFT + FIRE |

## Reward

Each demon killed awards points scaled by the current wave number:

```text
points = (wave + 1) * 30
```

The base value is 30 points per demon on wave 0. Higher waves multiply the per-kill reward.

## Episode End

The episode ends when all lives are lost. A life is lost when an enemy bullet hits the cannon, or when the demon formation descends to cannon level.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Cannon | y = 185, x ∈ [8, 144] |
| Demon formation top | y = 30 (initial) |
| Demon grid | 3 rows x 6 columns |
| Row spacing | 20 px |
| Column spacing | 22 px |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/demon_attack-v0")
play("atari/demon_attack-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE |
| `→` / `D` | RIGHT |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
