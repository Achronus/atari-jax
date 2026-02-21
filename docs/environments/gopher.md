# Gopher

> ALE name: `gopher` · Game ID: `23`

Protect three carrots from a tunnelling gopher by shooting it when it surfaces and filling in the holes it digs in your garden.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for shooting the gopher and its offspring. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when all three carrots have been eaten by the gopher, detected when the 3-bit carrot bitmask reaches zero.

## Lives

Lives are the number of surviving carrots (0 to 3), encoded as the popcount of a 3-bit bitmask in RAM. The episode ends when all three carrots are gone.
