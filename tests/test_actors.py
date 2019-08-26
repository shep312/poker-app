from poker.actors import Player


def test_high_card():

    # Create a player instance and add some cards
    player = Player()
    cards = [
        dict(suit=1, value=8, name='dummy_name'),
        dict(suit=2, value=9, name='dummy_name_2'),
        dict(suit=3, value=9, name='dummy_name_3')
    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['High card', 'high_card'] == 9
    assert player.hand_score.loc['High card', 'present']
    assert player.hand_score.loc['High card', 'required_cards'] == [[2, 9]]
    assert len(player.hand_score.loc['High card', 'required_cards']) == 1


def test_pair():

    # Create a player instance and add some cards
    player = Player()
    cards = [
        dict(suit=1, value=9, name='dummy_name'),
        dict(suit=2, value=9, name='dummy_name_2')
    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['Pair', 'high_card'] == 9
    assert player.hand_score.loc['Pair', 'present']
    assert player.hand_score.loc['Pair', 'required_cards'] == \
        [[1, 9], [2, 9]]

    extra_card = dict(suit=3, value=9, name='dummy_card_3')
    player.hand = player.hand.append(extra_card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)
    player.determine_hand()
    assert player.hand_score.loc['Pair', 'present']
    assert len(player.hand_score.loc['Pair', 'required_cards']) == 2


def test_two_pair():
    player = Player()
    cards = [
        dict(suit=1, value=8, name='dummy_name'),
        dict(suit=2, value=8, name='dummy_name_2'),
        dict(suit=1, value=3, name='dummy_name_3'),
        dict(suit=2, value=3, name='dummy_name_4')
    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['Two pairs', 'high_card'] == 8.03
    assert player.hand_score.loc['Two pairs', 'present']
    assert player.hand_score.loc['Two pairs', 'required_cards'] == \
        [[1, 8], [2, 8], [1, 3], [2, 3]]


def test_three_of_a_kind():
    player = Player()
    cards = [
        dict(suit=1, value=8, name='dummy_name'),
        dict(suit=2, value=8, name='dummy_name_2'),
        dict(suit=3, value=8, name='dummy_name_3'),
        dict(suit=2, value=3, name='dummy_name_4')
    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['Three of a kind', 'high_card'] == 8
    assert player.hand_score.loc['Three of a kind', 'present']
    assert player.hand_score.loc['Three of a kind', 'required_cards'] == \
        [[1, 8], [2, 8], [3, 8]]


def test_straight():
    player = Player()
    cards = [
        dict(suit=1, value=2, name='dummy_name'),
        dict(suit=2, value=3, name='dummy_name_2'),
        dict(suit=3, value=4, name='dummy_name_3'),
        dict(suit=2, value=5, name='dummy_name_4'),
        dict(suit=2, value=14, name='dummy_name_5')
    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['Straight', 'high_card'] == 5
    assert player.hand_score.loc['Straight', 'present']
    assert sorted(player.hand_score.loc['Straight', 'required_cards']) == \
        sorted([[1, 2], [2, 3], [3, 4], [2, 5], [2, 14]])

    player = Player()
    cards = [
        dict(suit=1, value=9, name='dummy_name'),
        dict(suit=1, value=10, name='dummy_name'),
        dict(suit=2, value=11, name='dummy_name_2'),
        dict(suit=3, value=12, name='dummy_name_3'),
        dict(suit=2, value=13, name='dummy_name_4'),
        dict(suit=2, value=14, name='dummy_name_5')
    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['Straight', 'high_card'] == 14
    assert player.hand_score.loc['Straight', 'present']
    assert player.hand_score.loc['Straight', 'required_cards'] == \
        [[1, 10], [2, 11], [3, 12], [2, 13], [2, 14]]


def test_flush():
    player = Player()
    cards = [
        dict(suit=1, value=2, name='dummy_name'),
        dict(suit=1, value=3, name='dummy_name_2'),
        dict(suit=1, value=4, name='dummy_name_3'),
        dict(suit=1, value=5, name='dummy_name_4'),
        dict(suit=1, value=6, name='dummy_name_4')

    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['Flush', 'high_card'] == 6
    assert player.hand_score.loc['Flush', 'present']
    assert player.hand_score.loc['Flush', 'required_cards'] == \
        [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6]]


def test_full_house():
    player = Player()
    cards = [
        dict(suit=1, value=2, name='dummy_name'),
        dict(suit=2, value=2, name='dummy_name_2'),
        dict(suit=3, value=3, name='dummy_name_3'),
        dict(suit=4, value=3, name='dummy_name_4'),
        dict(suit=1, value=3, name='dummy_name_4')

    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['Full house', 'high_card'] == 3.02
    assert player.hand_score.loc['Full house', 'present']
    assert sorted(player.hand_score.loc['Full house', 'required_cards']) == \
        sorted([[1, 2], [2, 2], [3, 3], [4, 3], [1, 3]])


def test_four_of_a_kind():
    player = Player()
    cards = [
        dict(suit=1, value=3, name='dummy_name'),
        dict(suit=2, value=3, name='dummy_name_2'),
        dict(suit=3, value=3, name='dummy_name_3'),
        dict(suit=4, value=3, name='dummy_name_4')

    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['Four of a kind', 'high_card'] == 3
    assert player.hand_score.loc['Four of a kind', 'present']
    assert sorted(player.hand_score.loc['Four of a kind', 'required_cards']) == \
        sorted([[1, 3], [2, 3], [3, 3], [4, 3]])


def test_straight_flush():
    player = Player()
    cards = [
        dict(suit=1, value=2, name='dummy_name'),
        dict(suit=1, value=3, name='dummy_name_2'),
        dict(suit=1, value=4, name='dummy_name_3'),
        dict(suit=1, value=5, name='dummy_name_4'),
        dict(suit=1, value=14, name='dummy_name_5')
    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['Straight flush', 'high_card'] == 5
    assert player.hand_score.loc['Straight flush', 'present']
    assert sorted(player.hand_score.loc['Straight flush', 'required_cards']) == \
        sorted([[1, 2], [1, 3], [1, 4], [1, 5], [1, 14]])


def test_royal_flush():
    player = Player()
    cards = [
        dict(suit=1, value=10, name='dummy_name'),
        dict(suit=1, value=11, name='dummy_name_2'),
        dict(suit=1, value=12, name='dummy_name_3'),
        dict(suit=1, value=13, name='dummy_name_4'),
        dict(suit=1, value=14, name='dummy_name_5')
    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)

    player.determine_hand()
    assert player.hand_score.loc['Royal flush', 'high_card'] == 14
    assert player.hand_score.loc['Royal flush', 'present']
    assert sorted(player.hand_score.loc['Royal flush', 'required_cards']) == \
        sorted([[1, 10], [1, 11], [1, 12], [1, 13], [1, 14]])


def test_remove_card():
    player = Player()
    cards = [
        dict(suit=1, value=10, name='dummy_name'),
        dict(suit=1, value=11, name='dummy_name_2')
    ]
    for card in cards:
        player.hand = player.hand.append(card, ignore_index=True)
    player.hand.reset_index(drop=True, inplace=True)
    player._remove_cards([[1, 10]])
    assert player.hand.shape[0] == 1
    assert 10 not in player.hand['value']


def test_fold():
    player = Player()
    assert not player.folded
    player.fold()
    assert player.folded


