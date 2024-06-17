from enum import Enum


class FruitClass(Enum):
    PEACH      = 0
    PEAR       = 1
    BANANA     = 2
    STRAWBERRY = 3
    APPLE      = 4
    SAPODILLA  = 5
    MANGO      = 6
    TOMATO     = 7

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
