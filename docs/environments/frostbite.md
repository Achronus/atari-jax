# Frostbite

> ALE name: `frostbite` · Game ID: `22`

Jump across rows of floating ice floes to collect ice blocks and build an igloo, while avoiding polar bears, fish, and crabs in the Arctic.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for landing on ice floes and for collecting fish. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the lives count reaches zero while the game's active-play flag is set, preventing false terminals on the title screen.

## Lives

Lives are stored as a zero-based count in the low nibble of a RAM byte (displayed lives = RAM nibble + 1). A life is lost by touching a polar bear, crab, or falling into the water.
