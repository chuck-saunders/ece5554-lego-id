from enum import Enum


class LightingConditions(Enum):
    DirectOverhead = 1,
    DirectOblique = 2,
    Indirect = 3,
    Low = 4


class BackgroundConditions(Enum):
    MatteWhite = 1,
    SemiGlossBlack = 2,
    LightWoodgrain = 3,
    DarkWoodgrain = 4,
    LightFabric = 5,
    DarkFabric = 6,
