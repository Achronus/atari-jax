# Enduro

> ALE name: `enduro` · Game ID: `19`

Race a car through changing day and night conditions, passing a required number of cars each day to continue to the next.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the number of cars passed per step, computed as a cumulative total across all days. Day 1 requires passing 200 cars; subsequent days require 300. The per-step reward reflects the incremental progress toward each day's quota.

## Episode End

The episode ends when the game-over flag is set, indicating the player failed to meet the day-1 quota before time ran out.

## Lives

No lives system. Enduro ends when the player fails to pass the daily car quota rather than losing lives.
