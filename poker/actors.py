import pandas as pd
import numpy as np
from poker.utils import poker_hands_rank, stage_names, \
    draws_remaining_at_stage, possible_straights


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
        assert not any(self.hand.duplicated()), 'player has duplicate cards'

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
            self._add_hand_to_player('High card')
            self.hand_score.loc['High card', 'high_card'] = \
                self.hand['value'].max()
        else:
            self.hand_score.loc['High card', 'probability_of_occurring'] = 1

        # Pair
        value_counts = self.hand['value'].value_counts()
        if any(value_counts == 2):
            self._add_hand_to_player('Pair')
            self.hand_score.loc['Pair', 'high_card'] = \
                value_counts[value_counts == 2].idxmax()
        else:
            # For each single card, there should be 3 others out there with
            # its value. 
            n_potential_cards = 3 * sum(value_counts == 1)
            p_pair = n_potential_cards * p_card
            # Also the chance that any of the future cards could
            # be pairs. Each number has a raw prob of 4/52. This needs to be 
            # reduced to reflect that not all cards remain (multiply by p_card),
            # then it needs to happen twice.
            if n_draws_remaining >= 2:
                p_pair += (p_card * 4 / 52)**2 
            self.hand_score.loc['Pair', 'probability_of_occurring'] = p_pair

        # Two pair
        if sum(value_counts == 2) == 2:
            self._add_hand_to_player('Two pairs')
            self.hand_score.loc['Two pairs', 'high_card'] = \
                value_counts[value_counts == 2].idxmax()
        else:
            # If there is already a pair, then the probability of another pair
            # given the remaining singles. 
            if sum(value_counts == 2) == 1:
                p_two_pair = 3 * sum(value_counts == 1) * p_card
            # If not, the probability of any one of our singles becoming
            # a pair up to a max of 2 pairs
            else:
                p_two_pair = (3 * p_card)**max(sum(value_counts == 1), 2)
            # If two pairs appear outside the current hand
            if n_draws_remaining >= 4:
                p_two_pair += 4 * p_card * 3 * p_card
            self.hand_score.loc['Two pairs', 'probability_of_occurring'] = \
                p_two_pair

        # Three of a kind
        if any(value_counts == 3):
            self._add_hand_to_player('Three of a kind')
            self.hand_score.loc['Three of a kind', 'high_card'] = \
                value_counts[value_counts == 3].idxmax()
        else:
            # For the single cards, need the probability of getting one of 
            # 3 cards followed by the probability of one of 2 cards.
            # For the double cards, one of 2 just once
            p_three = 3 * p_card * 2 * p_card * sum(value_counts==1)
            p_three += 2 * sum(value_counts==2) * p_card
            # Also the prob of one happening with future , unconnected cards
            if n_draws_remaining >= 3:
                p_three += (p_card * 4 / 52)**3 
            self.hand_score.loc['Three of a kind', 'probability_of_occurring'] \
                = p_three

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
            # Straight is tricky to work out so going for a brute force check.
            # Import a list of all the possible straights, then check the
            # probability of each one given the cards we have. Sum those
            # probabilities for the probability of any straight.
            p_straight = 0
            present_cards = set(aces_high_hand['value'].tolist()
                                + aces_low_hand['value'].tolist())
            for straight in possible_straights:
                p_this_straight = np.prod(np.array(
                    [1 if card in present_cards else 4 * p_card
                     for card in straight]
                ))
                p_straight += p_this_straight
            # Straight could occur in other five cards
            if n_draws_remaining >= 5:
                # TODO work this out (probably insignificant)
                p_straight += 0
            self.hand_score.loc['Straight', 'probability_of_occurring'] = \
                p_straight

        # Flush
        suit_counts = self.hand['suit'].value_counts()
        if any(suit_counts >= 5):
            flushed_suit = suit_counts[suit_counts >= 5].idxmax()
            self._add_hand_to_player('Flush')
            self.hand_score.loc['Flush', 'high_card'] = \
                self.hand.loc[self.hand['suit'] == flushed_suit, 'value'].max()
        else:
            p_flush = 0
            if n_draws_remaining >= 4:
                p_flush += sum(p_flush == 1) * ((13 * p_card) / 52)**4

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

    def _add_hand_to_player(self, hand):
        """
        If the appropriate conditions are met, add this hand to a players
        hand_score property

        Args:
            hand (str): String indicating the hand that is present
        """
        self.hand_score.loc[hand, 'present'] = True
        self.hand_score.loc[hand, 'probability_of_occurring'] = 1


class Opponent(Player):
    pass


class User(Player):
    pass
