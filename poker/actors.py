import pandas as pd
import numpy as np
from poker.utils import poker_hands_rank, stage_names, \
    draws_remaining_at_stage, possible_straights, SUITS, VALUES, \
    possible_full_houses


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

        # Determine how many card draws are left at this stage of the game
        stage = stage_names.get(self.hand.shape[0])
        n_draws_remaining = draws_remaining_at_stage[stage]  # type: int

        # Distance from dealer's left
        distance_from_dealers_left = self.table_position - 1
        if distance_from_dealers_left == -1:
            distance_from_dealers_left = n_players

        # Determine number of cards left in the deck for probability calculations
        n_cards_in_deck = len(deck)
        n_cards_unknown = 52 - self.hand.shape[0]

        # Probability that a card is still in the deck at this stage
        # I.E. Not in another player's hand
        p_card_in_deck = n_cards_in_deck / 52

        # Turn this probability into the probability that any given card
        # will be drawn during the remainder of the game.
        p_card = n_draws_remaining / 52

        # Need to account
        # for decreasing probability in the later stages of the game at the
        # early stages due to other players picking up cards.
        # Also need to account for the players position - dealer's left
        # gets first cards etc.
        # if stage == 'hole':
        #     # 3 draws from the current deck at the flop
        #     p_card = 3 * p_card_in_deck / n_cards_in_deck
        #     # Following this, 3 cards fewer in the deck available for the
        #     # turn. Also as many cards fewer as positions player is from the
        #     # dealer
        #     p_card += (n_cards_in_deck - 3) / 52 \
        #         / (n_cards_in_deck - 3 - distance_from_dealers_left)
        #     # Following this, 3 cards fewer in the deck from the flop,
        #     # plus 1 * n_players fewer from the turn
        #     p_card_drawn = (n_cards_in_deck - 3 - n_players) / 52 \
        #          / (n_cards_in_deck - 3 - n_players - distance_from_dealers_left)
        #     p_card += p_card_drawn
        # elif stage == 'turn':
        #     p_card = p_card_in_deck \
        #         / (n_cards_in_deck - distance_from_dealers_left)
        #     p_card_drawn = (n_cards_in_deck - n_players) / 52 \
        #         / (n_cards_in_deck - n_players - distance_from_dealers_left)
        #     p_card += p_card_drawn
        # elif stage == 'river':
        #     p_card = p_card_in_deck \
        #         / (n_cards_in_deck - distance_from_dealers_left)
        # else:
        #     raise ValueError('Stage %s not recognised' % stage)

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
            # its value. To get the probability of getting any of these cards,
            # first we get the probability of the sequence of not getting
            # any. The probability of our own pair (requires at least one
            # of our pocket cards) is then 1 - the sequence prob
            n_single_cards_in_hand = sum(value_counts == 1)
            n_cards_to_avoid = n_single_cards_in_hand * 3
            sequence = [(n_cards_unknown - n_cards_to_avoid - i)
                        / (n_cards_unknown - i)
                        for i in range(n_draws_remaining)]
            p_own_pair = 1 - np.prod(np.array(sequence))
            # Also the chance that any of the future cards could
            # be pairs. Again, easier to get sequence of it not happening
            # Sequence: draw a card (1), draw a card of different value,
            # draw a card of different value again...
            if n_draws_remaining >= 2:
                sequence = [1] + [(n_cards_unknown - 3 - i) / (n_cards_unknown - i)
                                  for i in range(n_draws_remaining - 1)]
                p_shared_pair = 1 - np.prod(np.array(sequence))
            else:
                p_shared_pair = 0
            p_pair = 1 - ((1 - p_own_pair) * (1 - p_shared_pair))
            self.hand_score.loc['Pair', 'probability_of_occurring'] = p_pair

        # Two pair
        if sum(value_counts == 2) >= 2:
            self._add_hand_to_player('Two pairs')
            self.hand_score.loc['Two pairs', 'high_card'] = \
                value_counts[value_counts == 2].idxmax()
        else:
            # If there is already a pair, then the probability of another pair
            # given the remaining singles. 
            if sum(value_counts == 2) == 1:
                if any(value_counts == 1):
                    n_cards_to_avoid = 3 * sum(value_counts == 1)
                    sequence = [(n_cards_unknown - n_cards_to_avoid - i)
                                / (n_cards_unknown - i)
                                for i in range(n_draws_remaining)]
                    p_two_pair_inc_pairs = 1 - np.prod(np.array(sequence))
                else:
                    if n_draws_remaining >= 2:
                        sequence = [1] + [(n_cards_unknown - 3 - i) / (n_cards_unknown - i)
                                          for i in range(1, n_draws_remaining)]
                        assert len(sequence) == n_draws_remaining
                        p_shared_pair = 1 - np.prod(np.array(sequence))
                    else:
                        p_shared_pair = 0
                    p_two_pair_inc_pairs = p_shared_pair
            else:
                p_two_pair_inc_pairs = 0

            # If not, the probability of any one of our singles becoming
            # a pair up to a max of 2 pairs
            n_single_cards_in_hand = sum(value_counts == 1)
            n_cards_to_avoid = n_single_cards_in_hand * 3

            # Sequence of no pairs off current singles
            sequence = [(n_cards_unknown - n_cards_to_avoid - i)
                        / (n_cards_unknown - i)
                        for i in range(n_draws_remaining)]
            sequence = np.array(sequence, ndmin=2)

            # Only get one pair. Start off with an square array describing
            # each possible sequence in which this happens
            sequences = np.zeros([n_draws_remaining, n_draws_remaining])
            for i in range(sequences.shape[0]):
                # Give every option the probability of no pair this draw
                for j in range(sequences.shape[1]):
                    sequences[i, j] = (n_cards_unknown - n_cards_to_avoid - j) \
                        / (n_cards_unknown - j)
                # For one draw per sequence, throw in a pair draw
                sequences[i, i] = n_cards_to_avoid / (n_cards_unknown - i)

            # Append the no pairs sequence
            sequences = np.concatenate([sequences, sequence], axis=0)

            # Get the probability that any of these sequences happen,
            # and then 1 - that is the probability of the alternative:
            # two pairs happen
            p_two_pair_off_singles = 1 - sequences.prod(axis=1).sum()

            # If two pairs appear outside the current hand
            if n_draws_remaining >= 4:

                # Sequences with only one pair or no pairs
                sequences = np.zeros([n_draws_remaining, n_draws_remaining])
                for i in range(sequences.shape[0]):
                    # Give every option the probability of no pair this draw
                    for j in range(1, sequences.shape[1]):
                        if i + 1 == j:
                            # For one draw per sequence, throw in a pair draw
                            sequences[i, j] = 3 / (n_cards_unknown - j)
                        else:
                            sequences[i, j] = (n_cards_unknown - 3 - j) \
                                              / (n_cards_unknown - j)

                    # Since were not fussed which pair, set the first p of
                    # every sequence to be 1
                    sequences[i, 0] = 1

                p_shared_two_pair = 1 - sequences.prod(axis=1).sum()

            else:
                p_shared_two_pair = 0

            # Actually probability of two pairs is 1 - the probability
            # of none ot the above paths to a two pair happening.
            # Has to be this and not an addition because each option
            # is not mutually exclusive
            p_two_pair = 1 - (
                (1 - p_shared_two_pair)
                * (1 - p_two_pair_inc_pairs)
                * (1 - p_two_pair_off_singles)
            )
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
            if n_draws_remaining >= 2:

                # Sequence of no matching cards are drawn to any on the current
                # singles
                sequence = [(n_cards_unknown - 3 - i)
                            / (n_cards_unknown - i)
                            for i in range(n_draws_remaining)]
                sequence = np.array(sequence, ndmin=2)

                # Sequences of just one matching card for each single
                sequences = np.zeros([n_draws_remaining, n_draws_remaining])
                for i in range(sequences.shape[0]):
                    # Give every option the probability of no match this draw
                    for j in range(sequences.shape[1]):
                        sequences[i, j] = (n_cards_unknown - 3 - j) \
                            / (n_cards_unknown - j)
                    # For one draw per sequence, throw in a match
                    sequences[i, i] = 3 / (n_cards_unknown - i)

                # Combine these sequences for all non-three draws
                sequences = np.concatenate([sequence, sequences], axis=0)

                # Probability that one of these sequences happens - giving
                # the probability that a given one of our singles gets to
                # triple
                p_not_three_for_each_single = sequences.prod(axis=1).sum()

                # Combined probability that this happens for any of the single
                # cards - non mutually exclusive
                p_own_three_from_one = 1 - \
                    p_not_three_for_each_single ** sum(value_counts == 1)
            else:
                p_own_three_from_one = 0

            # Prob of an existing pair becoming three
            if n_draws_remaining >= 1:

                # Sequence of no matching cards are drawn to any on the current
                # singles
                sequence = [(n_cards_unknown - 2 - i)
                            / (n_cards_unknown - i)
                            for i in range(n_draws_remaining)]
                sequence = np.array(sequence, ndmin=2)
                p_not_three_for_each_double = sequence.prod()

                # Combined probability that this happens for any of the single
                # cards - non mutually exclusive
                p_own_three_from_pair = 1 - \
                    p_not_three_for_each_double ** sum(value_counts == 2)

            else:
                p_own_three_from_pair = 0

            # Also the prob of one happening with future, unconnected cards
            if n_draws_remaining >= 3:

                # Sequences with only one match or no matches
                sequences = np.zeros([n_draws_remaining, n_draws_remaining])
                for i in range(sequences.shape[0]):
                    # Give every option the probability of no pair this draw
                    for j in range(1, sequences.shape[1]):
                        if i + 1 == j:
                            # For one draw per sequence, throw in a match draw
                            sequences[i, j] = 3 / (n_cards_unknown - j)
                        elif j <= i + 1:
                            # If we're pre-first match, 3 cards to avoid
                            sequences[i, j] = (n_cards_unknown - 3 - j) \
                                              / (n_cards_unknown - j)
                        else:
                            # If we're post, only 2 cards to avoid
                            sequences[i, j] = (n_cards_unknown - 2 - j) \
                                              / (n_cards_unknown - j)

                    # Since were not fussed which pair, set the first p of
                    # every sequence to be 1
                    sequences[i, 0] = 1

                # Account for fact this could occur for any of the values
                # we don't have
                p_not_shared_three = sequences.prod(axis=1).sum()
                p_shared_three = 1 - p_not_shared_three

            else:
                p_shared_three = 0

            p_three = 1 - (
                (1 - p_own_three_from_one)
                * (1 - p_own_three_from_pair)
                * (1 - p_shared_three)
            )
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
            straight_non_probs, straight_probs = [], []
            present_cards = set(self.hand['value'].tolist())

            # Sequence required to not get any particular card needed
            # for a straight
            sequence = [(n_cards_unknown - 4 - i) / (n_cards_unknown - i)
                        for i in range(n_draws_remaining)]
            sequence = np.array(sequence)

            # For each straight, get the probability of getting each card
            # needed, which are independent events
            for straight in possible_straights:

                # How many cards do we not have that are needed to complete
                # this straight
                n_cards_needed = \
                    sum([card not in present_cards for card in straight])

                # Now need to add the probabilities of each of the options for
                # not getting this straight

                # The probability of not getting a single required card
                sequence = [(n_cards_unknown - 4 - i) / (n_cards_unknown - i)
                            for i in range(n_draws_remaining)]
                p_no_needed_cards = np.prod(np.array(sequence))

                # The probability of getting just one required card
                # |   card   | not card | not card |
                # |--------------------------------|
                # | not card |   card   | not card |
                # |--------------------------------|
                # | not card | not card |   card   |
                sequences = np.zeros([n_draws_remaining, n_draws_remaining])
                for i in range(sequences.shape[0]):
                    # Give every option the probability of no match this draw
                    for j in range(sequences.shape[1]):
                        sequences[i, j] = (n_cards_unknown - 4 - j) \
                                          / (n_cards_unknown - j)
                    # For one draw per sequence, throw in a match
                    sequences[i, i] = 4 / (n_cards_unknown - i)
                p_one_needed_card = sequences.prod(axis=1).sum()

                # The probability of getting just two required cards
                # |   card1  | card2 | not card1 and |
                # |          |       | not card 2    |
                # |--------------------------------|
                # | not card |   card   | not card |
                # |--------------------------------|
                # | not card | not card |   card   |
                sequences = np.zeros([n_draws_remaining, n_draws_remaining])
                for i in range(sequences.shape[0]):
                    # Give every option the probability of no match this draw
                    for j in range(sequences.shape[1]):
                        sequences[i, j] = (n_cards_unknown - 4 - j) \
                                          / (n_cards_unknown - j)
                    # For one draw per sequence, throw in a match
                    sequences[i, i] = 4 / (n_cards_unknown - i)
                p_two_needed_cards = sequences.prod(axis=1).sum()

                # Different probability calculations depending on how many cards
                # needed to complete the straight
                if n_cards_needed > n_draws_remaining:
                    p_not_this_straight = 1

                elif n_cards_needed == 1:
                    p_not_this_straight = p_no_needed_cards

                elif n_cards_needed == 2:
                    p_not_this_straight = \
                        p_no_needed_cards + p_one_needed_card

                elif n_cards_needed == 3:
                    pass

                elif n_cards_needed == 4:
                    pass

                elif n_cards_needed == 5:
                    pass

                else:
                    raise ValueError('Unacceptable number of cards needed'
                                     'for straight: %s' % str(n_cards_needed))



                # p_each_card_in_straight = [1 if card in present_cards
                #                            else 1 - np.prod(sequence)
                #                            for card in straight]
                # p_this_straight = np.prod(np.array(p_each_card_in_straight))
                # # Record array of probabilities of not getting each straight
                # straight_non_probs.append(1 - p_this_straight)
                straight_probs.append(1 - 5)

            # How to convert these into a single probability? Some straights
            # Are mutually exclusive and some aren't...
            p_straight = sum(straight_probs)
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
                p_flush += sum(suit_counts == 1) \
                           * ((39 / 52) ** n_draws_remaining) ** (5 - 1)
                p_flush += sum(suit_counts == 2) \
                           * ((39 / 52) ** n_draws_remaining) ** (5 - 2)
                p_flush += sum(suit_counts == 3) \
                           * ((39 / 52) ** n_draws_remaining) ** (5 - 3)
            elif n_draws_remaining == 2:
                p_flush += sum(suit_counts == 3) \
                           * ((39 / 52) ** n_draws_remaining) ** (5 - 3)
                p_flush += sum(suit_counts == 4) \
                           * ((39 / 52) ** n_draws_remaining) ** (5 - 4)
            elif n_draws_remaining == 1:
                p_flush += sum(suit_counts == 4) \
                           * ((39 / 52) ** n_draws_remaining) ** (5 - 4)
            self.hand_score.loc['Flush', 'probability_of_occurring'] = \
                p_flush

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
            # Loop through all full houses and sum probabilities of
            # each one occurring
            p_of_not_each_house = []
            present_values = self.hand['value'].values.tolist()
            for full_house in possible_full_houses:
                p_each_card = []
                for card in full_house:
                    if card in present_values:
                        p_each_card.append(1)
                        present_values.remove(card)
                    else:
                        n_potential_cards = 4 - sum(self.hand['value'] == card)
                        p_each_card.append(
                            1 - (((52 - n_potential_cards) / 52) ** n_draws_remaining)
                        )
                p_dont_get_cards = [1 - prob for prob in p_each_card]
                p_of_not_each_house.append(np.prod(np.array(p_dont_get_cards)))
            p_full_house = 1 - np.prod(np.array(p_of_not_each_house))

            self.hand_score.loc['Full house', 'probability_of_occurring'] = \
                p_full_house

        # Four of a kind
        if any(value_counts == 4):
            self.hand_score.loc['Four of a kind', 'present'] = True
            self.hand_score.loc['Four of a kind', 'probability_of_occurring'] = 1
            self.hand_score.loc['Four of a kind', 'high_card'] = \
                value_counts[value_counts == 4].index.max()
        else:
            # For the single cards, need the probability of getting one of
            # 3 cards, followed by the probability of one of 2 cards,
            # followed by the probability of one of 1 cards.
            # For the double cards, one of 2 just once.
            # For the tripe cards, one of just 1 once.
            n_potential_cards = 3 * sum(value_counts == 1)
            p_four = \
                (1 - (((52 - n_potential_cards) / 52) ** n_draws_remaining)) \
                ** 3
            n_potential_cards = 2 * sum(value_counts == 2)
            p_four += \
                (1 - (((52 - n_potential_cards) / 52) ** n_draws_remaining)) \
                ** 2
            n_potential_cards = sum(value_counts == 3)
            p_four += \
                (1 - (((52 - n_potential_cards) / 52) ** n_draws_remaining))
            # Also the prob of one happening with future, unconnected cards
            if n_draws_remaining >= 4:
                p_four += 1 - ((48 / 52) ** (n_draws_remaining - 4))
            self.hand_score.loc['Four of a kind', 'probability_of_occurring'] \
                = p_four

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
        else:
            p_aces_high_sf = 0
            for straight in possible_straights:
                for suit in SUITS.keys():
                    suited_cards = \
                        aces_high_hand[aces_high_hand['suit'] == suit]
                    p_aces_high_sf += np.prod(np.array(
                        [1 if card in suited_cards['value'] else p_card
                         for card in straight]
                    ))

            p_royal_flush = 0
            for suit in SUITS.keys():
                suited_cards = \
                    aces_high_hand[aces_high_hand['suit'] == suit]
                p_royal_flush += np.prod(np.array(
                    [1 if card in suited_cards['value'] else p_card
                     for card in [10, 11, 12, 13, 14]]
                ))

            p_aces_low_sf = 0
            for straight in possible_straights:
                for suit in SUITS.keys():
                    suited_cards = \
                        aces_low_hand[aces_low_hand['suit'] == suit]
                    p_aces_low_sf += np.prod(np.array(
                        [1 if card in suited_cards['value'] else p_card
                         for card in straight]
                    ))

            self.hand_score.loc['Royal flush', 'probability_of_occurring'] = \
                p_royal_flush

            self.hand_score.loc['Straight flush', 'probability_of_occurring'] = \
                p_aces_low_sf + p_aces_high_sf

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
