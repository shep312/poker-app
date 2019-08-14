import functools
import numpy as np


def prob_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        p = func(*args, **kwargs)
        assert p >= 0 and p <= 1, 'Probability calculated incorrectly'
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