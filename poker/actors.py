import pandas as pd
import numpy as np
from poker.utils import poker_hands_rank, stage_names, draws_remaining_at_stage


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

    def determine_hand(self, n_players, deck):
        """
        Determine what hands are present in a player's hand
        and therefore how strong it is
    
        Args:
             n_players (int): Number of players in game
             deck (list): Cards remaining in the deck
        """
        assert isinstance(n_players, int), 'n_players must be a integer'
        assert self.hand.shape[0] <= 7, 'player has more than 7 cards'

        # Determine number of cards left in the deck for probability calculations
        n_cards_in_deck = len(deck)
        n_cards_with_opponents = 52 - n_cards_in_deck - self.hand.shape[0]

        # Probability any given card is in the deck
        p_card_in_deck = (n_cards_in_deck - n_cards_with_opponents) / 52

        # Probability that card can be drawn in current stage
        stage = stage_names.get(self.hand.shape[0])
        n_draws_remaining = draws_remaining_at_stage[stage]
        p_card_drawn = p_card_in_deck / n_cards_in_deck
        p_card = n_draws_remaining * p_card_drawn

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
            # For each single card, there should be 3 others out there with
            # its value
            n_potential_cards = 3 * sum(value_counts == 1)
            self.hand_score.loc['Pair', 'probability_of_occurring'] = \
                n_potential_cards * p_card

        # Two pair
        if sum(value_counts == 2) == 2:
            self.hand_score.loc['Two pairs', 'present'] = True
            self.hand_score.loc['Two pairs', 'probability_of_occurring'] = 1
            self.hand_score.loc['Two pairs', 'high_card'] = \
                value_counts[value_counts == 2].idxmax()
        else:
            # If there is already a pair, then the probability of another pair
            # given the remaining singles. If not, the probability of any
            # one of our singles becoming a pair up to a max of 2 pairs
            if sum(value_counts == 2) == 1:
                p = 3 * sum(value_counts == 1) * p_card
            else:
                p = (3 * p_card)**max(sum(value_counts == 1), 2)
            self.hand_score.loc['Pair', 'probability_of_occurring'] = p

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
        aces_high_hand = self.hand.copy()
        aces_low_hand = aces_high_hand.copy()
        aces_low_hand['value'] = aces_low_hand['value'].replace(14, 1)

        def calc_streak(hand):
            hand = hand.sort_values(by='value')
            hand['diff'] = hand['value'].diff()
            hand['not_linked'] = (hand['diff'] != 1).cumsum()
            hand['streak'] = hand.groupby('not_linked').cumcount()
            return hand

        aces_high_hand = calc_streak(aces_high_hand)
        aces_low_hand = calc_streak(aces_low_hand)

        straight_type = None
        if aces_high_hand['streak'].max() >= 4:
            self.hand_score.loc['Straight', 'present'] = True
            self.hand_score.loc['Straight', 'probability_of_occurring'] = 1
            self.hand_score.loc['Straight', 'high_card'] = \
                aces_high_hand.loc[aces_high_hand['streak'] == 4, 'value'].values
            straight_type = 'aces_high'
        elif aces_low_hand['streak'].max() >= 4:
            self.hand_score.loc['Straight', 'present'] = True
            self.hand_score.loc['Straight', 'probability_of_occurring'] = 1
            self.hand_score.loc['Straight', 'high_card'] = \
                aces_low_hand.loc[aces_low_hand['streak'] == 4, 'value'].values
            straight_type = 'aces_low'
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
        def check_straight_type(hand):
            straight_link = \
                hand.loc[hand['streak'] == 4, 'not_linked']
            straight_cards = \
                hand[hand['not_linked'] == straight_link.max()]
            if straight_cards['suit'].nunique() == 1 \
                    and straight_cards['value'].max() == 14:
                return 'Royal flush'
            elif straight_cards['suit'].nunique() == 1:
                return 'Straight flush'
            else:
                return None

        if straight_type == 'aces_high':
            if check_straight_type(aces_high_hand) == 'Straight flush':
                self.hand_score.loc['Straight flush', 'present'] = True
                self.hand_score.loc['Straight flush',
                                    'probability_of_occurring'] = 1
                self.hand_score.loc['Straight flush', 'high_card'] = \
                    self.hand_score.loc['Straight', 'high_card']
            else:
                # TODO probability calc
                pass
            if check_straight_type(aces_high_hand) == 'Royal flush':
                self.hand_score.loc['Royal flush', 'present'] = True
                self.hand_score.loc['Royal flush', 'probability_of_occurring'] = 1
                self.hand_score.loc['Royal flush', 'high_card'] = \
                    self.hand_score.loc['Straight', 'high_card']
            else:
                # TODO probability calc
                pass
        elif straight_type == 'aces_low':
            if check_straight_type(aces_low_hand) == 'Straight flush':
                self.hand_score.loc['Straight flush', 'present'] = True
                self.hand_score.loc['Straight flush',
                                    'probability_of_occurring'] = 1
                self.hand_score.loc['Straight flush', 'high_card'] = \
                    self.hand_score.loc['Straight', 'high_card']
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
