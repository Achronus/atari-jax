# Skiing

> ALE name: `skiing` · Game ID: `44`

Ski down a slalom course, passing through gates as quickly as possible; the reward is a time penalty so lower elapsed time is better.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is negative on every step, equal to minus the amount of time elapsed. It is computed as `previous elapsed time minus current elapsed time`, so the agent receives a penalty for every centisecond that passes. Finishing quickly minimises the total penalty.

## Episode End

The episode ends when the run is complete, detected when the terminal flag byte equals `0xFF`.

## Lives

No lives system. Skiing is a single timed run with no lives concept.
