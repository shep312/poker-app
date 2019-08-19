import pytest
from pandas import DataFrame
from poker.utils import get_card_name, ncr
from poker.probabilities import get_boolean_sequence, \
    sum_multiple_sequence_probabilities, calculate_pair_prob


def test_pair_prob():
    hand = DataFrame({
        'value': [1, 2],
        'suit': [0, 0]
    })
    p = calculate_pair_prob(hand, 50, 1)
    assert round(p, 5) == round(6 / 50, 5)

    p = calculate_pair_prob(hand, 50, 2)
    assert round(p, 5) > round(6 / 50, 5)

    assert calculate_pair_prob(hand, 45, 3) \
        > calculate_pair_prob(hand, 50, 3)

    hand = DataFrame({
        'value': [1, 2, 3],
        'suit': [0, 0, 0]
    })
    p = calculate_pair_prob(hand, 50, 1)
    assert round(p, 5) == round(9 / 50, 5)


def test_two_pair_prob():
    hand = DataFrame({
        'value': [1, 2, 2],
        'suit': [0, 0, 0]
    })
    p = calculate_pair_prob(hand, 50, 1)
    assert round(p, 5) == round(3 / 50, 5)

    hand = DataFrame({
        'value': [1, 2, 2, 3],
        'suit': [0, 0, 0, 0]
    })
    p = calculate_pair_prob(hand, 50, 1)
    assert round(p, 5) == round(6 / 50, 5)


def test_get_boolean_sequence():

    # Check can't handle case where there are more events than draws
    assert not get_boolean_sequence(4, 3), \
        'Should return false in more events than draws'

    # Over 5 draws
    with pytest.raises(ValueError):
        get_boolean_sequence(6, 6)

    sequence = get_boolean_sequence(3, 5)
    assert sequence.shape[1] == 5, 'Not enough draws in sequences'
    assert all(sequence.sum(axis=1) == 3), 'Not right number of true events'

    sequence = get_boolean_sequence(1, 2)
    assert sequence.shape[1] == 2, 'Not enough draws in sequences'
    assert all(sequence.sum(axis=1) == 1), 'Not right number of true events'


def test_sum_multiple_sequence_probabilities():
    p_not_desired_sequence = sum_multiple_sequence_probabilities(
        n_cards_unknown=40,
        n_cards_left=10,
        n_draws=2,
        same_card_type=False,
        n_cards_needed=3
    )
    assert p_not_desired_sequence == 1
