import numpy as np
from poker.game import Game


def test_init():
    n_players = 5
    game = Game(n_players)
    assert len(game.players) == n_players, 'Wrong number of players assigned'

    dealership = [player.is_dealer for player in game.players]
    assert sum(dealership) == 1, 'Should only be one dealer'


def test_prepare_deck():
    n_players = 5
    game = Game(n_players)
    assert len(game.deck) == 52, 'Too many cards to start'

    ex_cards = [[1, 9], [3, 4], [2, 2]]
    game.prepare_deck(excluded_cards=ex_cards)
    assert len(game.deck) == (52 - len(ex_cards)), \
        'Card exclusion when preparing deck not working'


def test_deal_hole():
    n_players = 6
    game = Game(n_players)
    game.deal_hole()
    player_n_cards = np.array([len(player.hand) for player in game.players])
    assert all(player_n_cards == 2), \
        'Not all players have two cards after dealing hole cards'
    assert len(game.deck) + sum(player_n_cards) == 52, \
        'Some cards have gone missing after dealing hole'


def test_deal_card():
    n_players = 6
    game = Game(n_players)
    game.deal_hole()
    deck_size = len(game.deck)
    community_card_size = len(game.community_cards)

    game.deal_card()
    new_deck_size = len(game.deck)
    new_community_card_size = len(game.community_cards)

    deck_difference = new_deck_size - deck_size
    community_difference = new_community_card_size - community_card_size
    assert community_difference == -deck_difference, \
        'Cards have gone missing on dealing to community'

    game.deal_card(game.players[0])
    assert len(game.players[0].hole) == 3, \
        'Player has not received dealt card'


def test_simulate():
    pass


def test_determine_winner():
    game = Game(n_players=2)
    
    player_1_cards = [
        dict(suit=1, value=8, name='dummy_name'),
        dict(suit=2, value=8, name='dummy_name_2'),
        dict(suit=1, value=3, name='dummy_name_3'),
        dict(suit=2, value=3, name='dummy_name_4')
    ]
    for card in player_1_cards:
        game.players[0].hand = \
            game.players[0].hand.append(card, ignore_index=True)
    game.players[0].hand.reset_index(drop=True, inplace=True)
    
    player_2_cards = [
        dict(suit=1, value=8, name='dummy_name'),
        dict(suit=2, value=8, name='dummy_name_2'),
        dict(suit=1, value=2, name='dummy_name_3'),
        dict(suit=2, value=2, name='dummy_name_4')
    ]
    for card in player_2_cards:
        game.players[1].hand = \
            game.players[1].hand.append(card, ignore_index=True)
    game.players[1].hand.reset_index(drop=True, inplace=True)
    
    for player in game.players:
        player.determine_hand()
    print(game.players[0].hand_score)
    winners = game.determine_winner()
    assert game.players[0] in winners, 'Winner not in winners list'
    assert game.players[1] not in winners, 'Loser in winners list'
    
