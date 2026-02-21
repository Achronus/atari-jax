# Bowling

> ALE name: `bowling` · Game ID: `10`

Bowl a standard ten-frame game, aiming for strikes and spares to accumulate the highest possible score up to a perfect 300.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned based on the number of pins knocked down per roll, with bonus points for strikes and spares. Score is stored as packed BCD across two RAM bytes.

## Episode End

The episode ends when the ten-frame game is complete, detected when the round counter exceeds `0x10`.

## Lives

No lives system. Bowling is a single-game session with no lives concept; the episode always ends after ten frames.
