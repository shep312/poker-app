from app import app
from poker.game import Game
from flask import render_template


@app.route('/')
@app.route('/index')
def index():
    run = Game(n_players=4, simulation_iterations=5, parallelise=False)
    run.deal_hole()
    run.simulate()
    return render_template('index.html', 
                           hand=run.user.hand.to_html(), 
                           hand_probs=run.user.hand_score.to_html(),
                           win_prob=100 * run.user.win_probability)