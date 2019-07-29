import pandas as pd
import numpy as np
from poker.utils import poker_hands_rank, stage_names


class Player:

    def __init__(self):
        self.table_position = 0
        self.is_dealer = False
        self.is_small_blind = False
        self.is_big_blind = False
        self.folded = False
        self.hole = pd.DataFrame(columns=['suit', 'value', 'name'])
        self.hand = pd.DataFrame(columns=['suit', 'value', 'name'])
        self.hand_score = pd.DataFrame({
            'hand': list(poker_hands_rank.keys()),
            'hand_rank': list(poker_hands_rank.values()),
            'present': np.zeros([len(poker_hands_rank), ], dtype=bool),
            'probability_of_occurring': np.zeros([len(poker_hands_rank), ]),
            'high_card': np.zeros([len(poker_hands_rank), ], dtype=int)
        }).set_index('hand')
        self.best_hand = None
        self.best_hand_high_card = 0
        self.best_hand_numeric = 0

    def determine_hand(self, n_players):
        """
        Determine what hands are present in a player's hand
        and therefore how strong it is
    
        Args:
             hand (pandas.DataFrame): The hand as a dataframe
             n_players (int): Number of players in game    
        """
        assert isinstance(n_players, int), 'n_players must be a integer'
        assert self.hand.shape[0] <= 7, 'player has more than 7 cards'

        # Determine number of cards left in the deck for probability calculations
        stage = stage_names.get(self.hand.shape[0], 'unknown')
        if stage != 'not_started':
            n_hole_cards = 2
            n_community_cards = max(self.hand.shape[0] - n_hole_cards, 0)
            cards_in_deck = 52 - n_community_cards - n_players * 2
        else:
            cards_in_deck = 52

        # Run checks to see what hands are currently present
        # High card
        if self.hand.shape[0]:
            self.hand_score.loc['High card', 'present'] = True
            self.hand_score.loc['High card', 'probability_of_occurring'] = 1
            self.hand_score.loc['High card', 'high_card'] = \
                self.hand['value'].max()
        else:
            self.hand_score.loc['High card', 'probability_of_occurring'] = 1

        # Pair
        value_counts = self.hand['value'].value_counts()
        if any(value_counts == 2):
            self.hand_score.loc['Pair', 'present'] = True
            self.hand_score.loc['Pair', 'probability_of_occurring'] = 1
            self.hand_score.loc['Pair', 'high_card'] = \
                value_counts[value_counts == 2].idxmax()
        else:
            # TODO probability calc
            pass

        # Two pair
        if sum(value_counts == 2) == 2:
            self.hand_score.loc['Two pairs', 'present'] = True
            self.hand_score.loc['Two pairs', 'probability_of_occurring'] = 1
            self.hand_score.loc['Two pairs', 'high_card'] = \
                value_counts[value_counts == 2].idxmax()
        else:
            # TODO probability calc
            pass

        # Three of a kind
        if any(value_counts == 3):
            self.hand_score.loc['Three of a kind', 'present'] = True
            self.hand_score.loc['Three of a kind', 'probability_of_occurring'] = 1
            self.hand_score.loc['Three of a kind', 'high_card'] = \
                value_counts[value_counts == 3].idxmax()
        else:
            # TODO probability calc
            pass

        # Straight
        # TODO need to account for the case where Ace is a 1
        sorted_hand = self.hand.sort_values(by='value')
        sorted_hand['diff'] = sorted_hand['value'].diff()
        sorted_hand['not_linked'] = (sorted_hand['diff'] != 1).cumsum()
        sorted_hand['streak'] = sorted_hand.groupby('not_linked').cumcount()
        if sorted_hand['streak'].max() >= 4:
            self.hand_score.loc['Straight', 'present'] = True
            self.hand_score.loc['Straight', 'probability_of_occurring'] = 1
            self.hand_score.loc['Straight', 'high_card'] = \
                sorted_hand.loc[sorted_hand['streak'] == 4, 'value'].values
        else:
            # TODO probability calc
            pass

        # Flush
        suit_counts = self.hand['suit'].value_counts()
        if any(suit_counts >= 5):
            flushed_suit = suit_counts[suit_counts >= 5].idxmax()
            self.hand_score.loc['Flush', 'present'] = True
            self.hand_score.loc['Flush', 'probability_of_occurring'] = 1
            self.hand_score.loc['Flush', 'high_card'] = \
                self.hand.loc[self.hand['suit'] == flushed_suit, 'value'].max()
        else:
            # TODO probability calc
            pass

        # Full house
        if any(value_counts == 2) and any(value_counts == 3):
            self.hand_score.loc['Full house', 'present'] = True
            self.hand_score.loc['Full house', 'probability_of_occurring'] = 1
            triple_value = value_counts[value_counts == 3].idxmax()
            double_value = value_counts[value_counts == 2].idxmax()
            # So 8s full of 3s would be 3.08
            self.hand_score.loc['Full house', 'high_card'] = \
                triple_value + double_value / 100
        else:
            # TODO probability calc
            pass

        # Four of a kind
        if any(value_counts == 4):
            self.hand_score.loc['Four of a kind', 'present'] = True
            self.hand_score.loc['Four of a kind', 'probability_of_occurring'] = 1
            self.hand_score.loc['Four of a kind', 'high_card'] = \
                value_counts[value_counts == 4].index.max()
        else:
            # TODO probability calc
            pass

        # Straight flush
        if self.hand_score.loc['Straight', 'present']:
            straight_link = \
                sorted_hand.loc[sorted_hand['streak'] == 4, 'not_linked']
            straight_cards = \
                sorted_hand[sorted_hand['not_linked'] == straight_link.max()]
            if straight_cards['suit'].nunique() == 1:
                self.hand_score.loc['Straight flush', 'present'] = True
                self.hand_score.loc['Straight flush',
                                    'probability_of_occurring'] = 1
                self.hand_score.loc['Straight flush', 'high_card'] = \
                    self.hand_score.loc['Straight', 'high_card']
        else:
            # TODO probability calc
            pass

        # Royal flush
        if self.hand_score.loc['Straight flush', 'present'] \
                and self.hand_score.loc['Flush', 'high_card'] == 14:
            self.hand_score.loc['Royal flush', 'present'] = True
            self.hand_score.loc['Royal flush', 'probability_of_occurring'] = 1
            self.hand_score.loc['Royal flush', 'high_card'] = \
                self.hand_score.loc['Straight flush', 'high_card']
        else:
            # TODO probability calc
            pass

        # Pick out best hand
        present_hands = self.hand_score[self.hand_score['present'] == True]
        self.best_hand = present_hands['hand_rank'].idxmax()
        self.best_hand_high_card = \
            present_hands.loc[self.best_hand, 'high_card']
        # Convert the hand to numeric rank
        self.best_hand_numeric = \
            present_hands.loc[self.best_hand, 'hand_rank'] \
            + self.best_hand_high_card / 100


class Opponent(Player):
    pass


class User(Player):
    pass
