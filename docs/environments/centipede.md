# Centipede

> ALE name: `centipede` · Game ID: `13`

Shoot a multi-segment centipede as it descends through a mushroom field, along with fleas, spiders, and scorpions.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for each centipede segment, mushroom, and other insect destroyed. Negative score deltas are clamped to zero to protect against a score-reset glitch.

## Episode End

The episode ends when a game-over bit flag is set in a RAM byte. A life is lost when a centipede segment or spider reaches and touches the player.

## Lives

Lives are encoded in a 3-bit field of a RAM byte (lives displayed = bits shifted and offset by 1). A life is lost each time the player is touched by an enemy.
