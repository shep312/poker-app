import pandas as pd
import numpy as np
from poker.utils import HANDS_RANK, CARDS_IN_HAND


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
            'hand': list(HANDS_RANK.keys()),
            'hand_rank': list(HANDS_RANK.values()),
            'present': np.zeros([len(HANDS_RANK), ], dtype=bool),
            'probability_of_occurring': np.zeros([len(HANDS_RANK), ]),
            'high_card': np.zeros([len(HANDS_RANK), ], dtype=int),
        }).set_index('hand')
        self.hand_score['required_cards'] = \
            np.empty((len(self.hand_score), 0)).tolist()
        self.hand_score_numeric = 0.0

    def determine_hand(self):
        """
        Determine what hands are present in a player's hand
        and therefore how strong it is
        """
        assert 2 <= self.hand.shape[0] <= 7, 'player has a bad number of cards'
        assert not any(self.hand.duplicated()), 'player has duplicate cards'

        # Run checks to see what hands are currently present
        # High card
        if self.hand.shape[0]:
            self._add_hand_to_player('High card')
            max_value = self.hand['value'].max()
            self.hand_score.loc['High card', 'high_card'] = max_value
            # Just need one of the cards with this value
            self.hand_score.at['High card', 'required_cards'] = \
                self.hand.loc[self.hand['value'] == max_value, ['suit', 'value']]\
                .values.tolist()[:1]
                
        # Pair
        value_counts = self.hand['value'].value_counts()
        if any(value_counts == 2):
            self._add_hand_to_player('Pair')
            pair_value = value_counts[value_counts == 2].index.max()
            self.hand_score.loc['Pair', 'high_card'] = pair_value
            # Need two of the cards with this value
            self.hand_score.at['Pair', 'required_cards'] = \
                self.hand.loc[self.hand['value'] == pair_value, ['suit', 'value']]\
                    .values.tolist()[:2]

        # Two pair
        if sum(value_counts == 2) >= 2:
            self._add_hand_to_player('Two pairs')
            pair_values = value_counts[value_counts == 2].index.tolist()
            self.hand_score.loc['Two pairs', 'high_card'] = max(pair_values)
            self.hand_score.at['Two pairs', 'required_cards'] = \
                self.hand.loc[self.hand['value'].isin(pair_values), ['suit', 'value']]\
                    .values.tolist()

        # Three of a kind
        if any(value_counts == 3):
            self._add_hand_to_player('Three of a kind')
            three_value = value_counts[value_counts == 3].index.max()
            self.hand_score.loc['Three of a kind', 'high_card'] = three_value
            self.hand_score.at['Three of a kind', 'required_cards'] = \
                self.hand.loc[self.hand['value'] == three_value, ['suit', 'value']]\
                    .values.tolist()[:3]

        # Straight
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
            high_card_idx = aces_high_hand['streak'].idxmax()
            high_card = aces_high_hand.loc[high_card_idx, 'value']
            self.hand_score.loc['Straight', 'high_card'] = high_card
            link_num = aces_high_hand.loc[high_card_idx, 'not_linked']
            self.hand_score.at['Straight', 'required_cards'] = \
                aces_high_hand.loc[aces_high_hand['not_linked'] == link_num,
                                   ['suit', 'value']].values.tolist()
            straight_type = 'aces_high'
        elif aces_low_hand['streak'].max() >= 4:
            self.hand_score.loc['Straight', 'present'] = True
            self.hand_score.loc['Straight', 'probability_of_occurring'] = 1
            high_card_idx = aces_low_hand['streak'].idxmax()
            high_card = aces_low_hand.loc[high_card_idx, 'value']
            self.hand_score.loc['Straight', 'high_card'] = high_card
            link_num = aces_low_hand.loc[high_card_idx, 'not_linked']
            self.hand_score.at['Straight', 'required_cards'] = \
                aces_low_hand.loc[aces_low_hand['not_linked'] == link_num,
                                  ['suit', 'value']].values.tolist()
            straight_type = 'aces_low'

        # Flush
        suit_counts = self.hand['suit'].value_counts()
        if any(suit_counts >= 5):
            flushed_suit = suit_counts[suit_counts >= 5].index.max()
            self._add_hand_to_player('Flush')
            self.hand_score.loc['Flush', 'high_card'] = \
                self.hand.loc[self.hand['suit'] == flushed_suit, 'value'].max()

        # Full house
        if any(value_counts == 2) and any(value_counts == 3):
            self.hand_score.loc['Full house', 'present'] = True
            self.hand_score.loc['Full house', 'probability_of_occurring'] = 1
            triple_value = value_counts[value_counts == 3].index.max()
            double_value = value_counts[value_counts == 2].index.max()
            # So 8s full of 3s would be 3.08
            self.hand_score.loc['Full house', 'high_card'] = \
                triple_value + double_value / 100

        # Four of a kind
        if any(value_counts == 4):
            self.hand_score.loc['Four of a kind', 'present'] = True
            self.hand_score.loc['Four of a kind', 'probability_of_occurring'] = 1
            self.hand_score.loc['Four of a kind', 'high_card'] = \
                value_counts[value_counts == 4].index.max()

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
            if check_straight_type(aces_high_hand) == 'Royal flush':
                self.hand_score.loc['Royal flush', 'present'] = True
                self.hand_score.loc['Royal flush', 'probability_of_occurring'] = 1
                self.hand_score.loc['Royal flush', 'high_card'] = \
                    self.hand_score.loc['Straight', 'high_card']

        elif straight_type == 'aces_low':
            if check_straight_type(aces_low_hand) == 'Straight flush':
                self.hand_score.loc['Straight flush', 'present'] = True
                self.hand_score.loc['Straight flush',
                                    'probability_of_occurring'] = 1
                self.hand_score.loc['Straight flush', 'high_card'] = \
                    self.hand_score.loc['Straight', 'high_card']

    def _add_hand_to_player(self, hand):
        """
        If the appropriate conditions are met, add this hand to a players
        hand_score property

        Parameters
        ----------
        hand : str
            String indicating the hand that is present
        """
        self.hand_score.loc[hand, 'present'] = True
        self.hand_score.loc[hand, 'probability_of_occurring'] = 1


class Opponent(Player):
    pass


class User(Player):

    def __init__(self):
        super().__init__()
        self.win_probability = 0
