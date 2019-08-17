import operator as op
from functools import reduce
from numpy import arange, array


# Define the possible hands and their ranks
poker_hands_rank = {
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

# Define the stage names during the game. Keys are the number of cards
# a player has at that stage
stage_names = {
    0: 'not_started',
    2: 'hole',
    5: 'flop',
    6: 'turn',
    7: 'river'
}

# Draws remaining at a given stage
draws_remaining_at_stage = {
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

# Possible straights for the probability calc
possible_straights = [arange(i, i + 5) for i in range(1, 11)]
possible_straights.append(array([14, 2, 3, 4, 5]))

# Possible full houses for the probability calc
possible_full_houses = [[i, i, i, j, j] for i in range(2, 14)
                        for j in range(2, 14) if i != j]


def get_card_name(card):
    """
    Convert the integer-based card into English.
    e.g. (0, 2) returns '2 of Spades'

    Args:
        card (tuple): tuple of two integers defining the suit and value
                      respectively

    Returns:
        A string describing the hand in English
    """
    return '{} of {}'.format(VALUES[card[1]], SUITS[card[0]])


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom
