

# Define the possible hands and their ranks
HANDS_RANK = {
    'High card': 0,
    'Pair': 1,
    'Two pairs': 2,
    'Three of a kind': 3,
    'Straight': 4,
    'Flush': 5,
    'Full house': 6,
    'Four of a kind': 7,
    'Straight flush': 8,
    'Royal flush': 9
}

# Number of cards required in each hand - used to get best 5 card hand per
# player
CARDS_IN_HAND = {
    'High card': 1,
    'Pair': 2,
    'Two pairs': 4,
    'Three of a kind': 3,
    'Straight': 5,
    'Flush': 5,
    'Full house': 5,
    'Four of a kind': 5,
    'Straight flush': 5,
    'Royal flush': 5
}

# Define the stage names during the game. Keys are the number of cards
# a player has at that stage
STAGE_NAMES = {
    0: 'not_started',
    2: 'hole',
    5: 'flop',
    6: 'turn',
    7: 'river'
}

# Draws remaining at a given stage
DRAWS_REMAINING_AT_STAGE = {
    'not_started': 7,
    'hole': 5,
    'flop': 2,
    'turn': 1,
    'river': 0
}

# Encoded card suits
SUITS = {
    0: 'Spades',
    1: 'Diamonds',
    2: 'Hearts',
    3: 'Clubs'
}

# Encoded card values
VALUES = {
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '10',
    11: 'Jack',
    12: 'Queen',
    13: 'King',
    14: 'Ace'
}


def get_card_name(card):
    """
    Convert the integer-based card into English.
    e.g. (0, 2) returns '2 of Spades'

    PARAMETERS
    ----------
    card : tuple
        tuple of two integers defining the suit and value respectively

    RETURNS
    -------
    A string describing the hand in English
    """
    return '{} of {}'.format(VALUES[card[1]], SUITS[card[0]])
