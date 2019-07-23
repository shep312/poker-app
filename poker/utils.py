import pandas as pd
import numpy as np


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


def get_card_name(card):
    """
    Convert the integer-based card into English.
    e.g. (0, 2) returns '2 of Spades'

    Args:
        card (tuple): tuple of two integers defining the suit and value
                      respectively

    Returns:
        A string describing the hand in English
    -------
    """
    suits = {
        0: 'Spades',
        1: 'Diamonds',
        2: 'Hearts',
        3: 'Clubs'
    }
    values = {
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
    return '{} of {}'.format(values[card[1]], suits[card[0]])


def determine_hand(hand, n_players):
    """
    Function to determine what hands are present in a player's hand
    and therefore how strong it is

    Args:
         hand (pandas.DataFrame): The hand as a dataframe
         n_players (int): Number of players in game

    Returns:

    """
    assert isinstance(hand, pd.DataFrame), 'Input must be a Dataframe'

    # Determine number of cards left in the deck for probability calculations
    stage = stage_names.get(hand.shape[0], 'unknown')
    if stage != 'not_started':
        n_hole_cards = 2
        n_community_cards = max(hand.shape[0] - n_hole_cards, 0)
        cards_in_deck = 52 - n_community_cards - n_players * 2
    else:
        cards_in_deck = 52

    # Initialise hand scoring dataframe
    hand_score = pd.DataFrame({
        'hand': list(poker_hands_rank.keys()),
        'hand_rank': list(poker_hands_rank.values()),
        'present': np.zeros([len(poker_hands_rank), ], dtype=bool),
        'probability_of_occurring': np.zeros([len(poker_hands_rank), ]),
        'high_card': np.zeros([len(poker_hands_rank), ], dtype=int)
    }).set_index('hand')

    # Run checks to see what hands are currently present
    # High card
    if hand.shape[0]:
        hand_score.loc['High card', 'present'] = True
        hand_score.loc['High card', 'probability_of_occurring'] = 1
        hand_score.loc['High card', 'high_card'] = hand['value'].max()
    else:
        hand_score.loc['High card', 'probability_of_occurring'] = 1

    # Pair
    if any(hand['value'].value_counts() == 2):
        hand_score.loc['Pair', 'present'] = True
        hand_score.loc['Pair', 'probability_of_occurring'] = 1
        hand_score.loc['Pair', 'high_card'] =
    else:
        #TODO probability calc
        pass

    # Two pair
    if sum(hand['value'].value_counts() == 2) == 2:
        hand_score.loc['Two pairs', 'present'] = True
        hand_score.loc['Two pairs', 'probability_of_occurring'] = 1
        hand_score.loc['Two pairs', 'high_card'] =
