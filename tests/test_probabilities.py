import pytest
from poker.probabilities import get_boolean_sequence, \
    sum_multiple_sequence_probabilities


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
