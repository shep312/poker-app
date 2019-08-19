import functools
import numpy as np
import pandas as pd
from poker.utils import possible_straights, possible_full_houses, SUITS, \
    VALUES, ncr


def prob_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # Get the number of draws remaining
        if 'n_draws_remaining' in kwargs:
            n_draws = kwargs['n_draws_remaining']
        else:
            n_draws = args[2]

        # If there is some left, called the prob calc, if not return 0 prob
        if n_draws:
            p = func(*args, **kwargs)
            assert 0 <= p <= 1, 'Probability calculated incorrectly'
        else:
            p = 0
        return p

    return wrapper


@prob_function
def calculate_pair_prob(hand, n_cards_unknown, n_draws):
    """
    Calculates the probability of getting a pair at any point during
    this game

    PARAMETERS
    ----------
    hand : pandas.Dataframe
        Players hand as a dataframe detailing which hands are held
    n_cards_unknown : int
        Number of cards in deck and opponents hands
    n_draws : int
        Remaining draws left in game

    RETURNS
    -------
    p_pair : float
        Probability that this player will get a pair at some point in the
        game
    """

    # Number of single cards in the hand currently gives the number
    # of possible cards that when drawn create a pair
    value_counts = hand['value'].value_counts()
    n_single_cards_in_hand = sum(value_counts == 1)
    n_desired_cards = n_single_cards_in_hand * 3

    # All the potential draws left at this stage
    all_options = ncr(n_cards_unknown, n_draws)

    # The number of options where one of the current singles becomes a pair
    pair_options = ncr(n_desired_cards, 1) \
        * ncr(n_cards_unknown - n_desired_cards, n_draws - 1)
    p_own_pair = pair_options / all_options

    # Number of options of getting a pair without using a card in the current
    # hand
    if n_draws >= 2:
        fresh_pair_options = ((13 - n_single_cards_in_hand) * ncr(4, 2)
            * ncr(n_cards_unknown - 4, n_draws - 2)) \
            + (n_single_cards_in_hand * ncr(3, 2)
            * ncr(n_cards_unknown - 3, n_draws - 2))
        p_shared_pair = fresh_pair_options / all_options
    else:
        p_shared_pair = 0

    # Convert to a single probability
    p_pair = 1 - ((1 - p_own_pair) * (1 - p_shared_pair))
    return p_pair


@prob_function
def calculate_two_pair_prob(hand, n_cards_unknown, n_draws):
    # If there is already a pair, then the probability of another pair
    # given the remaining singles. 
    value_counts = hand['value'].value_counts()
    all_options = ncr(n_cards_unknown, n_draws)

    n_singles = sum(value_counts == 1)
    n_pairs = sum(value_counts == 2)

    if n_pairs == 1:
        pass
    else:


        two_pair_options_fresh = \
            (13 - n_singles) * ncr(4, 2) * (12 - n_singles) * ncr(4, 2) \
            + ((n_draws - 4) / n_draws) * ncr(n_cards_unknown - 4, n_draws)

        two_pair_options_off_singles = n_singles * ncr(3, 1) \
            + ((n_draws - 2) / n_draws) * ncr(n_cards_unknown - 4, n_draws)

        two_pair_options_off_one_single = \
            (13 - n_singles) * ncr(4, 2) * ncr(3, 1) \
            + ((n_draws - 3) / n_draws) * ncr(n_cards_unknown - 4, n_draws)

    if sum(value_counts == 2) == 1:
        if any(value_counts == 1):
            n_cards_needed = 3 * sum(value_counts == 1)
            options = ncr(n_cards_needed, 1) \
                * ncr(n_cards_unknown - n_cards_needed, n_draws - 1)
            p_two_pair_off_pair_and_singles = options / all_options
        else:
            p_two_pair_off_pair_and_singles = 0
            options = 0



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

        p_not_this_straight = sum_multiple_sequence_probabilities(
            n_cards_unknown=n_cards_unknown,
            n_cards_left=4,
            n_draws=n_draws,
            same_card_type=False,
            n_cards_needed=n_cards_needed
        )
        straight_non_probs.append(p_not_this_straight)

    # P(straight) is then 1 - P(no straights)
    p_straight = 1 - np.array(straight_non_probs).prod()
    return p_straight

   
@prob_function
def calculate_flush_prob(hand, n_cards_unknown, n_draws):
    suit_counts = hand['suit'].value_counts()
    if n_draws >= 4:
        flush_non_probs = []

        for suit in SUITS.keys():

            n_suited = sum(hand['suit'] == suit)
            n_suits_left = 13 - n_suited
            n_cards_needed = 5 - n_suited

            p_not_this_flush = sum_multiple_sequence_probabilities(
                n_cards_unknown=n_cards_unknown,
                n_cards_left=n_suits_left,
                n_draws=n_draws,
                same_card_type=True,
                n_cards_needed=n_cards_needed
            )
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
            p_flush = (10 / n_cards_unknown) * (9 / (n_cards_unknown - 1))
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
    full_house_probs = []
    for full_house in possible_full_houses:
        full_house_values = list(set(full_house))

        n_of_value_a_needed = sum(hand['value'] == full_house_values[0])
        n_of_value_b_needed = sum(hand['value'] == full_house_values[1])
        n_of_value_a_left = 4 - n_of_value_a_needed
        n_of_value_b_left = 4 - n_of_value_b_needed

        all_options = ncr(n_cards_unknown, n_draws)
        desired_options = ncr(n_of_value_a_left, n_of_value_a_needed) \
            * ncr(n_of_value_b_left, n_of_value_b_needed)
        p_not_this_full_house = 1 - desired_options / all_options

        full_house_probs.append(p_not_this_full_house)

    p_full_house = 1 - np.array(full_house_probs).prod()
    return p_full_house


@prob_function
def calculate_four_of_a_kind_prob(hand, n_cards_unknown, n_draws):
    # TODO seems high
    value_counts = hand['value'].value_counts()

    if any(value_counts == 3):
        p_not_own_four_from_three = sum_multiple_sequence_probabilities(
            n_cards_unknown=n_cards_unknown,
            n_cards_left=1,
            n_draws=n_draws,
            same_card_type=True,
            n_cards_needed=1
        )
        p_own_four_from_three = 1 - p_not_own_four_from_three
    else:
        p_own_four_from_three = 0

    if any(value_counts == 2):
        p_not_own_four_from_two = sum_multiple_sequence_probabilities(
            n_cards_unknown=n_cards_unknown,
            n_cards_left=2,
            n_draws=n_draws,
            same_card_type=True,
            n_cards_needed=2
        )
        p_own_four_from_two = 1 - p_not_own_four_from_two
    else:
        p_own_four_from_two = 0

    if any(value_counts == 1):
        p_not_own_four_from_one = sum_multiple_sequence_probabilities(
            n_cards_unknown=n_cards_unknown,
            n_cards_left=3,
            n_draws=n_draws,
            same_card_type=True,
            n_cards_needed=3
        )
        p_own_four_from_one = 1 - p_not_own_four_from_one
    else:
        p_own_four_from_one = 0

    p_not_shared_four = sum_multiple_sequence_probabilities(
        n_cards_unknown=n_cards_unknown,
        n_cards_left=4,
        n_draws=n_draws,
        same_card_type=True,
        n_cards_needed=4
    )
    p_shared_four = 1 - p_not_shared_four

    p_four = 1 - (
        (1 - p_own_four_from_three)
        * (1 - p_own_four_from_two)
        * (1 - p_own_four_from_one)
        * (1 - p_shared_four)
    )
    return p_four


@prob_function
def calculate_straight_flush_prob(hand, n_cards_unknown, n_draws):
    straight_non_probs = []
    for straight in possible_straights:
        for suit in SUITS.keys():
            n_cards_needed = 5
            for _, card in hand.iterrows():
                if card['value'] in straight and card['suit'] == suit:
                    n_cards_needed -= 1
                # Handle Aces low straight
                elif card['value'] == 14 and 1 in straight \
                    and card['suit'] == suit:
                    n_cards_needed -= 1

            p_not_this_straight = sum_multiple_sequence_probabilities(
                n_cards_unknown=n_cards_unknown,
                n_cards_left=1,
                n_draws=n_draws,
                same_card_type=False,
                n_cards_needed=n_cards_needed
            )
            straight_non_probs.append(p_not_this_straight)
    p_royal_flush = 1 - np.array(straight_non_probs).prod()
    return p_royal_flush


@prob_function
def calculate_royal_flush_prob(hand, n_cards_unknown, n_draws):
    straight_non_probs = []
    straight = [10, 11, 12, 13, 14]
    for suit in SUITS.keys():
        n_cards_needed = 5
        for _, card in hand.iterrows():
            if card['value'] in straight and card['suit'] == suit:
                n_cards_needed -= 1

        p_not_this_straight = sum_multiple_sequence_probabilities(
            n_cards_unknown=n_cards_unknown,
            n_cards_left=1,
            n_draws=n_draws,
            same_card_type=False,
            n_cards_needed=n_cards_needed
        )
        straight_non_probs.append(p_not_this_straight)
    p_straight_flush = 1 - np.array(straight_non_probs).prod()
    return p_straight_flush


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
        return False

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

    else:
        raise ValueError('Unacceptable value for n_events')

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

    RETURNS
    -------
    sequences : np.array
        Sequence of the probabilities of each draw for all options where
        we don't get the desired outcome
    """

    # If no sequences present then give a probability of zeros
    if not isinstance(bol_sequences, np.ndarray):
        if not bol_sequences:
            return 0
        else:
            raise TypeError('Invalid type passed for bol_sequences')

    # If only one sequence make sure its 2D
    if len(bol_sequences.shape) == 1:
        bol_sequences = bol_sequences.reshape(1, -1)

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


def sum_multiple_sequence_probabilities(n_cards_unknown, n_cards_left, n_draws,
                                        same_card_type, n_cards_needed):
    """
    For complex sequences of probabilities (such as flushes and straights)
    I need to get the probability of them occurring by summing the
    probabilities of all the sequences of them not occurring.
    This function looks to containerise these calculations

    PARAMETERS
    ----------
    n_cards_unknown : int
        Number of cards still in the deck or in opponents hands
    n_cards_left : int
        Number of cards remaining that are desired in this sequence
    n_draws : int
        Number of draws left in this round
    same_card_type : bool
        Whether drawing this card reduces its probability of reappearing
        E.G. drawing a suit when looking for a flush would be true
    n_cards_needed : int
        How many desired cards would be needed

    RETURNS
    -------
    p_desired_cards_not_drawn : int
        Probability of all the sequences where we don't get n_cards_needed
    """
    if n_draws < n_cards_needed:
        return 1

    all_probs = []
    if n_cards_needed > 0:
        for i in range(n_cards_needed):

            bol_sequences = get_boolean_sequence(i, n_draws)
            sequences = apply_probability_to_sequence(
                bol_sequences=bol_sequences,
                n_cards_unknown=n_cards_unknown,
                n_desired_cards=n_cards_left,
                same_card_type=same_card_type
            )
            if isinstance(sequences, np.ndarray):
                p_this_many_cards = sequences.prod(axis=1).sum()
            else:
                p_this_many_cards = 0
            all_probs.append(p_this_many_cards)

    p_desired_cards_not_drawn = sum(all_probs[:n_cards_needed])
    return p_desired_cards_not_drawn
