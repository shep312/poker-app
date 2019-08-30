from app import app
from poker.game import Game


def test_app_game():

    app.config['game'] = Game(n_players=4,
                              simulation_iterations=5,
                              parallelise=False)

    users_cards = [[0, 6], [1, 6]]
    for card in users_cards:
        app.config['game'].deal_card(recipient=app.config['game'].user,
                                     card=card)
    app.config['game'].simulate()

    flop_cards = [[1, 11], [0, 4], [2, 5]]
    app.config['game'].deal_community(cards=flop_cards)
    app.config['game'].simulate()

    turn_card = [[2, 6]]
    app.config['game'].deal_community(cards=turn_card)
    app.config['game'].simulate()

    river_card = [[3, 9]]
    app.config['game'].deal_community(cards=river_card)
    app.config['game'].simulate()
