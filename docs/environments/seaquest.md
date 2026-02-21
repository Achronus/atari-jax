# Seaquest

> ALE name: `seaquest` · Game ID: `43`

Pilot a submarine through the ocean, shooting enemy fish and sharks while rescuing divers and surfacing regularly to replenish oxygen.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for shooting enemy fish and sharks and for successfully rescuing divers by surfacing with them. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when a terminal flag byte becomes non-zero. This is triggered when all lives are lost through running out of oxygen or being hit by an enemy.

## Lives

Lives are stored as a zero-based count (displayed lives = RAM value + 1). A life is lost by running out of oxygen without surfacing, being hit by an enemy, or letting the shark reach the submarine.
