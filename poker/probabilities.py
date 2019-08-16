import functools
import numpy as np
import pandas as pd
from poker.utils import possible_straights, possible_full_houses, SUITS


def prob_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        p = func(*args, **kwargs)
        assert 0 <= p <= 1, 'Probability calculated incorrectly'
        return p
    return wrapper


@prob_function
def calculate_pair_prob(hand, n_cards_unknown, n_draws):
    # For each single card, there should be 3 others out there with
    # its value. To get the probability of getting any of these cards,
    # first we get the probability of the sequence of not getting
    # any. The probability of our own pair (requires at least one
    # of our pocket cards) is then 1 - the sequence prob
    value_counts = hand['value'].value_counts()
    n_single_cards_in_hand = sum(value_counts == 1)
    n_cards_to_avoid = n_single_cards_in_hand * 3
    sequence = [(n_cards_unknown - n_cards_to_avoid - i) 
                / (n_cards_unknown - i) 
                for i in range(n_draws)]
    p_own_pair = 1 - np.prod(np.array(sequence))
    # Also the chance that any of the future cards could
    # be pairs. Again, easier to get sequence of it not happening
    # Sequence: draw a card (1), draw a card of different value,
    # draw a card of different value again...
    if n_draws >= 2:
        sequence = [1] + [(n_cards_unknown - 3 - i) / (n_cards_unknown - i)
                          for i in range(n_draws - 1)]
        p_shared_pair = 1 - np.prod(np.array(sequence))
    else:
        p_shared_pair = 0
    p_pair = 1 - ((1 - p_own_pair) * (1 - p_shared_pair))
    return p_pair


@prob_function
def calculate_two_pair_prob(hand, n_cards_unknown, n_draws):
    # If there is already a pair, then the probability of another pair
    # given the remaining singles. 
    value_counts = hand['value'].value_counts()
    if sum(value_counts == 2) == 1:
        if any(value_counts == 1):
            n_cards_to_avoid = 3 * sum(value_counts == 1)
            sequence = [(n_cards_unknown - n_cards_to_avoid - i)
                        / (n_cards_unknown - i)
                        for i in range(n_draws)]
            p_two_pair_inc_pairs = 1 - np.prod(np.array(sequence))
        else:
            if n_draws >= 2:
                sequence = [1] + [(n_cards_unknown - 3 - i) / (n_cards_unknown - i)
                                  for i in range(1, n_draws)]
                assert len(sequence) == n_draws
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
                for i in range(n_draws)]
    sequence = np.array(sequence, ndmin=2)

    # Only get one pair. Start off with an square array describing
    # each possible sequence in which this happens
    sequences = np.zeros([n_draws, n_draws])
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
    if n_draws >= 4:

        # Sequences with only one pair or no pairs
        sequences = np.zeros([n_draws, n_draws])
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
    return p_two_pair


@prob_function
def calculate_three_of_a_kind_prob(hand, n_cards_unknown, n_draws):
    # For the single cards, need the probability of getting one of 
    # 3 cards followed by the probability of one of 2 cards.
    # For the double cards, one of 2 just once
    value_counts = hand['value'].value_counts()
    if n_draws >= 2:

        # Sequence of no matching cards are drawn to any on the current
        # singles
        sequence = [(n_cards_unknown - 3 - i)
                    / (n_cards_unknown - i)
                    for i in range(n_draws)]
        sequence = np.array(sequence, ndmin=2)

        # Sequences of just one matching card for each single
        sequences = np.zeros([n_draws, n_draws])
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
    if n_draws >= 1:

        # Sequence of no matching cards are drawn to any on the current
        # singles
        sequence = [(n_cards_unknown - 2 - i)
                    / (n_cards_unknown - i)
                    for i in range(n_draws)]
        sequence = np.array(sequence, ndmin=2)
        p_not_three_for_each_double = sequence.prod()

        # Combined probability that this happens for any of the single
        # cards - non mutually exclusive
        p_own_three_from_pair = 1 - \
                                p_not_three_for_each_double ** sum(value_counts == 2)

    else:
        p_own_three_from_pair = 0

    # Also the prob of one happening with future, unconnected cards
    if n_draws >= 3:

        # Sequences with only one match or no matches
        sequences = np.zeros([n_draws, n_draws])
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
    return p_three


@prob_function
def calculate_straight_prob(hand, n_cards_unknown, n_draws):
    # P(any straight) = P(straight 1) + P(straight 2) - P(straight 1 and straight 2)
    # P(no straight)

    # Straight is tricky to work out so going for a brute force check.
    # Import a list of all the possible straights, then check the
    # probability of each one given the cards we have. Sum those
    # probabilities for the probability of any straight.
    straight_non_probs, straight_probs = [], []
    present_cards = set(hand['value'].tolist())

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
                    for i in range(n_draws)]
        p_no_needed_cards = np.prod(np.array(sequence))

        # The probability of getting just one required card
        # |   card   | not card | not card |
        # |--------------------------------|
        # | not card |   card   | not card |
        # |--------------------------------|
        # | not card | not card |   card   |
        sequences = np.zeros([n_draws, n_draws])
        for i in range(sequences.shape[0]):
            # Give every option the probability of no match this draw
            for j in range(sequences.shape[1]):
                sequences[i, j] = (n_cards_unknown - 4 - j) \
                                  / (n_cards_unknown - j)
            # For one draw per sequence, throw in a match
            sequences[i, i] = 4 / (n_cards_unknown - i)
        p_one_needed_card = sequences.prod(axis=1).sum()

        # The probability of getting just two required cards
        sequences = []
        for k in range(n_draws):
            tmp_sequence = np.zeros([n_draws - 1, n_draws])
            for i in range(tmp_sequence.shape[0]):
                # Give every option the probability of no match this draw
                for j in range(tmp_sequence.shape[1]):
                    tmp_sequence[i, j] = (n_cards_unknown - 4 - j) \
                                         / (n_cards_unknown - j)
                # For one draw per sequence, throw in a match
                tmp_sequence[i, k] = 4 / (n_cards_unknown - k)

            for j in range(tmp_sequence.shape[1] - 1):
                if j < k:
                    tmp_sequence[j, j] = 4 / (n_cards_unknown - j)
                else:
                    tmp_sequence[j, j + 1] = 4 / (n_cards_unknown - j - 1)

            sequences.append(tmp_sequence)
        two_sequences = pd.DataFrame(np.concatenate(sequences)) \
            .drop_duplicates().values
        p_two_needed_cards = two_sequences.prod(axis=1).sum()

        # The probability of getting just three required cards
        sequences = []
        for k in range(n_draws):
            tmp_sequence = np.zeros([n_draws - 1, n_draws])
            for i in range(tmp_sequence.shape[0]):
                # Give every option the probability of a match this draw
                for j in range(tmp_sequence.shape[1]):
                    tmp_sequence[i, j] = 4 / (n_cards_unknown - j)
                # For one draw per sequence, throw in not a match
                tmp_sequence[i, k] = (n_cards_unknown - 4 - k) \
                                     / (n_cards_unknown - k)

            for j in range(tmp_sequence.shape[1] - 1):
                if j < k:
                    tmp_sequence[j, j] = (n_cards_unknown - 4 - j) \
                                         / (n_cards_unknown - j)
                else:
                    tmp_sequence[j, j + 1] = (n_cards_unknown - 4 - j - 1) \
                                             / (n_cards_unknown - j - 1)

            sequences.append(tmp_sequence)
        three_sequences = pd.DataFrame(np.concatenate(sequences)) \
            .drop_duplicates().values
        p_three_needed_cards = three_sequences.prod(axis=1).sum()

        # The probability of getting four required cards
        sequences = np.zeros([n_draws, n_draws])
        for i in range(sequences.shape[0]):
            # Give every option the probability of no match this draw
            for j in range(sequences.shape[1]):
                sequences[i, j] = 4 / (n_cards_unknown - j)
            # For one draw per sequence, throw in a match
            sequences[i, i] = (n_cards_unknown - 4 - i) \
                              / (n_cards_unknown - i)
        p_four_needed_card = sequences.prod(axis=1).sum()

        # Different probability calculations depending on how many cards
        # needed to complete the straight. Sum all the circumstances
        # of this straight not happening
        if n_cards_needed > n_draws:
            p_not_this_straight = 1

        elif n_cards_needed == 1:
            p_not_this_straight = p_no_needed_cards

        elif n_cards_needed == 2:
            p_not_this_straight = \
                p_no_needed_cards + p_one_needed_card

        elif n_cards_needed == 3:
            p_not_this_straight = \
                p_no_needed_cards + p_one_needed_card \
                + p_two_needed_cards

        elif n_cards_needed == 4:
            p_not_this_straight = \
                p_no_needed_cards + p_one_needed_card \
                + p_two_needed_cards + p_three_needed_cards

        elif n_cards_needed == 5:
            p_not_this_straight = \
                p_no_needed_cards + p_one_needed_card \
                + p_two_needed_cards + p_three_needed_cards \
                + p_four_needed_card

        else:
            raise ValueError('Unacceptable number of cards needed'
                             'for straight: %s' % str(n_cards_needed))
        print(hand, straight, n_cards_needed, p_not_this_straight)
        straight_non_probs.append(p_not_this_straight)

    # P(straight) is then 1 - P(no straights)
    p_straight = 1 - np.array(straight_non_probs).prod()
    return p_straight

   
@prob_function
def calculate_flush_prob(hand, n_cards_unknown, n_draws):
    if n_draws >= 4:
        flush_non_probs = []

        for suit in SUITS.keys():

            n_suited = sum(hand['suit'] == suit)
            n_suits_left = 13 - n_suited
            n_cards_needed = 5 - n_suited

            # The probability of not getting a single required card
            sequence = [(n_cards_unknown - n_suits_left - i)
                        / (n_cards_unknown - i)
                        for i in range(n_draws)]
            p_no_needed_cards = np.prod(np.array(sequence))

            # The probability of getting just one required card
            bol_sequences = get_boolean_sequence(1, n_draws)
            one_sequences = apply_probability_to_sequence(
                bol_sequences=bol_sequences,
                n_cards_unknown=n_cards_unknown,
                n_desired_cards=n_suits_left,
                same_card_type=True
            )
            p_one_needed_card = one_sequences.prod(axis=1).sum()

            # The probability of getting just two required cards
            bol_sequences = get_boolean_sequence(2, n_draws)
            two_sequences = apply_probability_to_sequence(
                bol_sequences=bol_sequences,
                n_cards_unknown=n_cards_unknown,
                n_desired_cards=n_suits_left,
                same_card_type=True
            )
            p_two_needed_cards = two_sequences.prod(axis=1).sum()

            # The probability of getting just three required cards
            bol_sequences = get_boolean_sequence(3, n_draws)
            three_sequences = apply_probability_to_sequence(
                bol_sequences=bol_sequences,
                n_cards_unknown=n_cards_unknown,
                n_desired_cards=n_suits_left,
                same_card_type=True
            )
            p_three_needed_cards = three_sequences.prod(axis=1).sum()

            # The probability of getting four required cards
            bol_sequences = get_boolean_sequence(4, n_draws)
            four_sequences = apply_probability_to_sequence(
                bol_sequences=bol_sequences,
                n_cards_unknown=n_cards_unknown,
                n_desired_cards=n_suits_left,
                same_card_type=True
            )
            p_four_needed_card = four_sequences.prod(axis=1).sum()

            all_probs = [p_no_needed_cards, p_one_needed_card,
                         p_two_needed_cards, p_three_needed_cards,
                         p_four_needed_card]
            p_not_this_flush = sum(all_probs[:n_cards_needed])
            flush_non_probs.append(p_not_this_flush)

        # Can only get one flush at a time so take the most likely
        p_flush = 1 - np.array(flush_non_probs).max()

    elif n_draws == 2:
        if sum(suit_counts == 4):
            p_flush = 1 - (
                    ((n_cards_unknown - 9) / n_cards_unknown)
                    * ((n_cards_unknown - 8) / (n_cards_unknown - 1))
            )
        elif sum(suit_counts == 3):
            p_flush = (10 / n_cards_unknown) * (9 / n_cards_unknown - 1)
        else:
            p_flush = 0

    elif n_draws == 1:
        if sum(suit_counts == 4):
            p_flush = 9 / n_cards_unknown
        else:
            p_flush = 0

    else:
        raise ValueError('Unacceptable number of draws remaining '
                         'when calculating flush probability')

    return p_flush


@prob_function
def calculate_full_house_prob(hand, n_cards_unknown, n_draws):
    # Loop through all full houses and sum probabilities of
    # each one occurring
    p_of_not_each_house = []
    present_values = hand['value'].values.tolist()
    for full_house in possible_full_houses:
        p_each_card = []
        for card in full_house:
            if card in present_values:
                p_each_card.append(1)
                present_values.remove(card)
            else:
                n_potential_cards = 4 - sum(self.hand['value'] == card)
                p_each_card.append(
                    1 - (((52 - n_potential_cards) / 52) ** n_draws)
                )
        p_dont_get_cards = [1 - prob for prob in p_each_card]
        p_of_not_each_house.append(np.prod(np.array(p_dont_get_cards)))
    p_full_house = 1 - np.prod(np.array(p_of_not_each_house))
    return p_full_house


@prob_function
def calculate_four_of_a_kind_prob(hand, n_cards_unknown, n_draws):
    pass


@prob_function
def calculate_straight_flush_prob(hand, n_cards_unknown, n_draws):
    pass


@prob_function
def calculate_royal_flush_prob(hand, n_cards_unknown, n_draws):
    pass


def get_boolean_sequence(n_events, n_draws):
    """
    Calculates the possible sequences for a number of specific events
    to occur given the remaining draws

    PARAMETERS
    ----------
    n_events : int
        The number of events, or 'Trues', in each sequence
    n_draws : int
        Number of draws left in game - the length of the sequence

    RETURNS
    -------
    sequences : np.array
        Boolean array of all possible sequences of shape:
        [n possible sequences, n draws remaining]
    """

    # Check the sequence is even possible
    if n_draws < n_events:
        raise ValueError('Not enough draws left for this event')

    if n_events == 0:
        sequences = np.array([False for i in range(n_draws)])

    elif n_events == 1:
        sequences = np.zeros([n_draws, n_draws], dtype=bool)
        for i in range(sequences.shape[0]):
            sequences[i, i] = True

    elif n_events == 2:
        sequence_list = []
        for k in range(n_draws):
            tmp_sequence_bol = np.zeros([n_draws - 1, n_draws], dtype=bool)
            for i in range(tmp_sequence_bol.shape[0]):
                # For one draw per sequence, throw in a match
                tmp_sequence_bol[i, k] = True

            for j in range(tmp_sequence_bol.shape[1] - 1):
                if j < k:
                    tmp_sequence_bol[j, j] = True
                else:
                    tmp_sequence_bol[j, j + 1] = True

            sequence_list.append(tmp_sequence_bol)
        sequences = pd.DataFrame(np.concatenate(sequence_list)) \
            .drop_duplicates().values

    elif n_events == 3:
        sequences = []
        for k in range(n_draws):
            tmp_sequence_bol = np.ones([n_draws - 1, n_draws], dtype=bool)
            for i in range(tmp_sequence_bol.shape[0]):
                # For one draw per sequence, throw in a non-match
                tmp_sequence_bol[i, k] = False

            for j in range(tmp_sequence_bol.shape[1] - 1):
                if j < k:
                    tmp_sequence_bol[j, j] = False
                else:
                    tmp_sequence_bol[j, j + 1] = False
            sequences.append(tmp_sequence_bol)
        sequences = pd.DataFrame(np.concatenate(sequences)) \
            .drop_duplicates().values

    elif n_events == 4:
        sequences = np.ones([n_draws, n_draws], dtype=bool)
        for i in range(sequences.shape[0]):
            sequences[i, i] = False

    elif n_events == 5:
        sequences = [True for i in range(n_draws)]
        sequences = np.array(sequences)

    return sequences.astype(np.float)


def apply_probability_to_sequence(bol_sequences, n_cards_unknown,
                                  n_desired_cards, same_card_type):
    """
    Takes an array of potential dealing sequences in Boolean form, and
    calculates the probability of each deal occuring given the number
    of cards in the deck at that point and the number of cards that we
    might want

    PARAMETERS
    ----------
    bol_sequences : np.array
        Boolean array of all possible sequences of shape:
        [n possible sequences, n draws remaining]
    n_cards_unknown : int
        The number of cards in the deck and opponents hands
    n_desired_cards : int
        The number of cards in deck that would consitute a true
    same_card_type : bool
        Whether successive trues require the same type of card and therefore
        probability should diminish as we go
    """
    sequences = bol_sequences.copy()
    for i in range(sequences.shape[0]):
        for j in range(sequences.shape[1]):
            n_to_remove = sum(bol_sequences[i, :j]) if same_card_type else 0

            # Probability if false
            if not bol_sequences[i, j]:
                sequences[i, j] = \
                    (n_cards_unknown - (n_desired_cards - n_to_remove) - j) \
                    / (n_cards_unknown - j)
            # Probability if true
            else:
                sequences[i, j] = \
                    (n_desired_cards - n_to_remove) \
                    / (n_cards_unknown - j)

    return sequences
