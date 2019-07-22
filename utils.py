def get_card_name(card):
    suits = {
        0: 'Spades',
        1: 'Diamonds',
        2: 'Hearts',
        3: 'Clubs'
    }
    values = {
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
        10: '10',
        11: 'Jack',
        12: 'Queen',
        13: 'King',
        14: 'Ace'
    }
    return '{} of {}'.format(suits[card[0]], values[card[1]])