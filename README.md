# Poker app

A web application designed to help in decision making for poker games.

A user can input the hole cards they've been dealt, and get expected
probabilities of each hand as well as a probability of winning.

This can then be updated at each stage of the game

The probabilities are estimated through Monte Carlo simulations. Opponent
players' hands and the remaining deck are randomised for each iteration
simulation.

## Example

Initial landing page prompts user to input hole cards:

![home page](https://github.com/shep312/poker-app/tree/development/app/docs/start.png)

After input, the app will carry out simulations and provide estimations of
the probability of each hand occurring and whether the user will win:

![flop page](https://github.com/shep312/poker-app/tree/development/app/docs/after_hole.png)
