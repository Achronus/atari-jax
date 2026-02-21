# Zaxxon

> ALE name: `zaxxon` · Game ID: `56`

Pilot a spacecraft through a fortress in a diagonal isometric view, shooting enemy ships and installations while managing your altitude.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for destroying enemy ships, fuel tanks, and installations; scores are multiples of 100. Score is stored as packed BCD across two RAM bytes.

## Episode End

The episode ends when all lives are exhausted, detected when the lower 3 bits of the lives RAM byte reach zero.

## Lives

Lives are stored as a 0-based count in the lower 3 bits of a RAM byte. A life is lost by crashing into terrain, walls, or obstacles, or by being hit by enemy fire.
