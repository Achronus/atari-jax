# Gravitar

> ALE name: `gravitar` · Game ID: `24`

Pilot a spaceship through solar systems under the influence of gravity, destroying enemy bases and fuel tanks to earn points and fuel.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for destroying enemy cannons, fuel tanks, and other hazards in each planetary system. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the game-over screen is shown, detected when the screen-state RAM byte is set to `0x01`. The title screen (value `0x00`) returns a pinned lives value of 6 to avoid false zero readings before the game starts.

## Lives

The player starts with several lives stored as a zero-based count (displayed lives = RAM value + 1). A life is lost when the ship crashes into terrain or is destroyed by enemy fire.
