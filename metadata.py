from enum import Enum


class FruitClass(Enum):
    PEACH      = 1
    PEAR       = 2
    BANANA     = 3
    APPLE      = 4
    STRAWBERRY = 5
    SAPODILLA  = 6
    MANGO      = 7
    TOMATO     = 8

FruitMetadata = {
    FruitClass.PEACH : {
        'string': 'Peach',
        'range': 0,

    },
    FruitClass.PEAR : {
        'string': 'Pear',
        'range': 0,

    },
    FruitClass.BANANA : {
        'string': 'Banana',
        'range': 0,

    },
    FruitClass.APPLE : {
        'string': 'Apple',
        'range': 20,

    },
    FruitClass.STRAWBERRY : {
        'string': 'Strawberry',
        'range': 0,

    },
    FruitClass.SAPODILLA : {
        'string': 'Sapodilla',
        'range': 16,

    },
    FruitClass.MANGO : {
        'string': 'Mango',
        'range': 22,

    },
    FruitClass.TOMATO : {
        'string': 'Tomato',
        'range': 20,

    },
}
