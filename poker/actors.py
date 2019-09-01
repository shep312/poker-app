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
        self.hand_scores_numeric = []
        self.best_five_card_hand = []
        self.n_cards_remaining = 5

    def determine_hand(self):
        """
        Determine what hands are present in a player's hand
        and therefore how strong it is
        """
        assert self.hand.shape[0] <= 7, 'player has a bad number of cards'
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
        if any(value_counts >= 2):
            self._add_hand_to_player('Pair')
            pair_value = value_counts[value_counts >= 2].index.max()
            self.hand_score.loc['Pair', 'high_card'] = pair_value
            # Need two of the cards with this value
            self.hand_score.at['Pair', 'required_cards'] = \
                self.hand.loc[self.hand['value'] == pair_value, ['suit', 'value']]\
                    .values.tolist()[:2]

        # Two pair
        if sum(value_counts == 2) >= 2:
            self._add_hand_to_player('Two pairs')
            # Take top two pairs
            pair_values = \
                value_counts[value_counts == 2].index\
                .sort_values(ascending=False).tolist()[:2]
            self.hand_score.loc['Two pairs', 'high_card'] = \
                pair_values[0] + pair_values[1] / 100
            self.hand_score.at['Two pairs', 'required_cards'] = \
                self.hand.loc[self.hand['value'].isin(pair_values), ['suit', 'value']]\
                    .values.tolist()

        # Three of a kind
        if any(value_counts >= 3):
            self._add_hand_to_player('Three of a kind')
            three_value = value_counts[value_counts == 3].index.max()
            self.hand_score.loc['Three of a kind', 'high_card'] = three_value
            self.hand_score.at['Three of a kind', 'required_cards'] = \
                self.hand.loc[self.hand['value'] == three_value, ['suit', 'value']]\
                    .values.tolist()[:3]

        # Straight
        def calc_streak(hand):
            hand = hand.sort_values(by='value')
            hand['diff'] = hand['value'].diff()
            hand['not_linked'] = (hand['diff'] != 1).cumsum()
            hand['streak'] = hand.groupby('not_linked').cumcount()
            return hand

        # Create different hands based on aces high and aces low.
        # Only make a difference between them if an ace is present to save
        # expensive calls to calc strea
        aces_high_hand = self.hand.copy()
        if 14 in aces_high_hand['value'].values:
            aces_low_hand = aces_high_hand.copy()
            aces_low_hand['value'] = aces_low_hand['value'].replace(14, 1)
            aces_high_hand = calc_streak(aces_high_hand)
            aces_low_hand = calc_streak(aces_low_hand)
        else:
            aces_high_hand = calc_streak(aces_high_hand)
            aces_low_hand = aces_high_hand.copy()

        straight_type = None
        if aces_high_hand['streak'].max() >= 4:
            self.hand_score.loc['Straight', 'present'] = True
            self.hand_score.loc['Straight', 'probability_of_occurring'] = 1
            high_card_idx = aces_high_hand['streak'].idxmax()
            high_card = aces_high_hand.loc[high_card_idx, 'value']
            self.hand_score.loc['Straight', 'high_card'] = high_card
            link_num = aces_high_hand.loc[high_card_idx, 'not_linked']
            # Take top 5 in best hand
            self.hand_score.at['Straight', 'required_cards'] = \
                aces_high_hand.loc[aces_high_hand['not_linked'] == link_num,
                                   ['suit', 'value']].values.tolist()[-5:]
            straight_type = 'aces_high'
        elif aces_low_hand['streak'].max() >= 4:
            self.hand_score.loc['Straight', 'present'] = True
            self.hand_score.loc['Straight', 'probability_of_occurring'] = 1
            high_card_idx = aces_low_hand['streak'].idxmax()
            high_card = aces_low_hand.loc[high_card_idx, 'value']
            self.hand_score.loc['Straight', 'high_card'] = high_card
            link_num = aces_low_hand.loc[high_card_idx, 'not_linked']
            # Create DF of required hands to convert aces back
            req_hands_df = aces_low_hand.loc[aces_low_hand['not_linked'] == link_num,
                                             ['suit', 'value']]
            req_hands_df['value'] = req_hands_df['value'].replace(1, 14)
            self.hand_score.at['Straight', 'required_cards'] = \
                req_hands_df.values.tolist()[-5:]
            straight_type = 'aces_low'

        # Flush
        suit_counts = self.hand['suit'].value_counts()
        if any(suit_counts >= 5):
            flushed_suit = suit_counts[suit_counts >= 5].index.max()
            self._add_hand_to_player('Flush')
            high_card = \
                self.hand.loc[self.hand['suit'] == flushed_suit, 'value'].max()
            self.hand_score.loc['Flush', 'high_card'] = high_card
            # TODO not sure which cards to require in case of over 5 matching
            # cards
            self.hand_score.at['Flush', 'required_cards'] = \
                self.hand.loc[self.hand['suit'] == flushed_suit,
                              ['suit', 'value']].values.tolist()[:5]

        # Full house
        if any(value_counts == 2) and any(value_counts == 3):
            self.hand_score.loc['Full house', 'present'] = True
            self.hand_score.loc['Full house', 'probability_of_occurring'] = 1
            triple_value = value_counts[value_counts == 3].index.max()
            double_value = value_counts[value_counts == 2].index.max()
            # So 8s full of 3s would be 3.08
            self.hand_score.loc['Full house', 'high_card'] = \
                triple_value + double_value / 100

            req_cards = \
                self.hand.loc[self.hand['value'] == triple_value,
                              ['suit', 'value']].values.tolist()[:3]\
                + self.hand.loc[self.hand['value'] == double_value,
                                ['suit', 'value']].values.tolist()[:2]
            self.hand_score.at['Full house', 'required_cards'] = req_cards

        # Four of a kind
        if any(value_counts == 4):
            self.hand_score.loc['Four of a kind', 'present'] = True
            self.hand_score.loc['Four of a kind', 'probability_of_occurring'] = 1
            four_value = value_counts[value_counts == 4].index.max()
            self.hand_score.loc['Four of a kind', 'high_card'] = four_value
            self.hand_score.at['Four of a kind', 'required_cards'] = \
                self.hand.loc[self.hand['value'] == four_value, ['suit', 'value']]\
                    .values.tolist()

        # Straight flush
        def check_straight_type(hand):
            straight_version = None
            straight_link = \
                hand.loc[hand['streak'].max(), 'not_linked']
            cards = \
                hand[hand['not_linked'] == straight_link.max()].copy()

            value_counts = cards['suit'].value_counts()
            if value_counts.max() >= 5 \
                    and cards['value'].max() == 14:
                straight_version = 'Royal flush'
                straight_suit = value_counts.idxmax()
                cards = \
                    cards[cards['suit'] == straight_suit]

            elif value_counts.max() >= 5:
                straight_version = 'Straight flush'
                straight_suit = value_counts.idxmax()
                cards = \
                    cards[cards['suit'] == straight_suit]

            cards.reset_index(drop=True, inplace=True)
            cards = cards[['suit', 'value']].iloc[:5, :]
            return straight_version, cards

        if straight_type == 'aces_high':
            straight_level, straight_cards = check_straight_type(aces_high_hand)
            if straight_level == 'Straight flush':
                self.hand_score.loc['Straight flush', 'present'] = True
                self.hand_score.loc['Straight flush',
                                    'probability_of_occurring'] = 1
                self.hand_score.loc['Straight flush', 'high_card'] = \
                    self.hand_score.loc['Straight', 'high_card']
                self.hand_score.at['Straight flush', 'required_cards'] = \
                    straight_cards.values.tolist()[-5:]
            if straight_level == 'Royal flush':
                self.hand_score.loc['Royal flush', 'present'] = True
                self.hand_score.loc['Royal flush', 'probability_of_occurring'] = 1
                self.hand_score.loc['Royal flush', 'high_card'] = \
                    self.hand_score.loc['Straight', 'high_card']
                self.hand_score.at['Royal flush', 'required_cards'] = \
                    straight_cards.values.tolist()[-5:]

        elif straight_type == 'aces_low':
            straight_level, straight_cards = check_straight_type(aces_low_hand)
            straight_cards['value'] = straight_cards['value'].replace(1, 14)
            if straight_level == 'Straight flush':
                self.hand_score.loc['Straight flush', 'present'] = True
                self.hand_score.loc['Straight flush',
                                    'probability_of_occurring'] = 1
                self.hand_score.loc['Straight flush', 'high_card'] = \
                    self.hand_score.loc['Straight', 'high_card']
                self.hand_score.at['Straight flush', 'required_cards'] = \
                    straight_cards.values.tolist()[-5:]
                    
    def get_best_five_card_hand(self):
        
        # Find the player's best hand
        present_hands = self.hand_score[self.hand_score['present']].copy()
        present_hands.sort_values(by=['hand_rank', 'high_card'], 
                                  ascending=[False, False],
                                  inplace=True)
        best_hand_idx = present_hands.index[0]
        best_hand_cards = present_hands.loc[best_hand_idx, 'required_cards']
        self.n_cards_remaining -= len(best_hand_cards)
        if self.n_cards_remaining < 0:
            print(self.n_cards_remaining)
            print(self.hand_score)
            print(best_hand_cards, best_hand_idx)
        assert self.n_cards_remaining >= 0, 'Remaining cards allowed has gone negative'
        
        # Assign hand score. E.G.:
        #    Full house of 2s full of 3s = 603.02
        #    High card of an ace = 14.0
        hand_score_numeric = \
            present_hands.loc[best_hand_idx, 'hand_rank'] * 100 \
            + present_hands.loc[best_hand_idx, 'high_card']
        self.hand_scores_numeric.append(hand_score_numeric)
        self.best_five_card_hand += best_hand_cards
        assert len(self.best_five_card_hand) <= 5, 'Too many cards in best hand'
            
        # Rescore without the best hand
        self._remove_cards(best_hand_cards)
        self._reset_hand_score()
        self.determine_hand()
        
        # Recurse until hands full
        if self.n_cards_remaining:
            self.get_best_five_card_hand()

    def fold(self):
        self.folded = True

    def _remove_cards(self, cards):
        for card in cards:
            mask = (self.hand['value'] == card[1]) \
                   & (self.hand['suit'] == card[0])
            self.hand = self.hand[~mask].reset_index(drop=True)

    def _reset_hand_score(self):
        self.hand_score = pd.DataFrame({
            'hand': list(HANDS_RANK.keys()),
            'hand_rank': list(HANDS_RANK.values()),
            'present': np.zeros([len(HANDS_RANK), ], dtype=bool),
            'probability_of_occurring': np.zeros([len(HANDS_RANK), ]),
            'high_card': np.zeros([len(HANDS_RANK), ], dtype=int),
        }).set_index('hand')
        self.hand_score['required_cards'] = \
            np.empty((len(self.hand_score), 0)).tolist()

    def _add_hand_to_player(self, hand):
        """
        If the appropriate conditions have been met, add this hand to a players
        hand_score property. Contingent on the player having enough cards
        left to include in their final 5 card hand.

        PARAMETERS
        ----------
        hand : str
            String indicating the hand that is present
        """
        if self.n_cards_remaining >= CARDS_IN_HAND[hand]:
            self.hand_score.loc[hand, 'present'] = True
            self.hand_score.loc[hand, 'probability_of_occurring'] = 1


class Opponent(Player):
    pass


class User(Player):

    def __init__(self):
        super().__init__()
        self.win_probability = 0
        self.draw_probability = 0
