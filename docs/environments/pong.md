# Pong

> ALE name: `pong` · Game ID: `36`

Volley a ball back and forth using a paddle against a CPU opponent; the first player to reach 21 points wins.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the change in score differential (player score minus CPU score) per step. Scoring a point yields +1; conceding a point yields -1. Scores are stored as raw integers (not BCD).

## Episode End

The episode ends when either player reaches 21 points.

## Lives

No lives system. Pong ends when a player reaches 21 points rather than through life loss.
