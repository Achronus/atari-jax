# Defender

> Game ID: `"atari/defender-v0"`

Protect humanoids on the ground from alien abductors. Your ship scrolls horizontally; shoot landers and mutants while keeping the humanoids alive. The episode ends when all humanoids have been taken.

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
| `3` | DOWN |
| `4` | THRUST (accelerate) |
| `5` | REVERSE (flip direction) |
| `6` | SMART BOMB |
| `7` | HYPERSPACE |
| `8` | FIRE + THRUST |

## Reward

| Event | Reward |
| --- | --- |
| Lander shot | +150 |
| Mutant shot | +150 |
| Humanoid rescued | +500 |

## Episode End

The episode ends when all humanoids have been abducted and lives reach zero.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Ground | y = 185 |
| Player initial y | y = 100 |
| Landers (world coords) | x ∈ [20, 620] |
| Humanoids (world coords) | x ∈ [30, 590], at ground level |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/defender-v0")
play("atari/defender-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE |
| `↑` / `W` | UP |
| `↓` / `S` | DOWN |
| `→` / `D` | THRUST |
| `←` / `A` | REVERSE |
| `Esc` / close window | Quit |
