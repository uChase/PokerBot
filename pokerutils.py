import itertools
import random
import random
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def card_to_encoder(cards):
    # Define the suits and ranks
    suits = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
    ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']

    # Create a mapping from card to index with ranks grouped together
    card_to_index = {f"{rank}{suit}": index for index, (rank, suit) in enumerate(itertools.product(ranks, suits))}

    # Initialize a 52-length binary array
    encoder = [0] * 52

    # Set the corresponding indices to 1 for each card
    for card in cards:
        encoder[card_to_index[card]] = 1

    return encoder


def encode_player_action_rl(action):

    actions = ['fold', 'c', 'raise']
    one_hot_encoded = [0] * len(actions)

    if action in actions:
        index = actions.index(action)
        one_hot_encoded[index] = 1

    return one_hot_encoded


def encode_player_action_lstm(action):

    actions = ['fold', 'check', 'call', 'bet', 'raise']
    one_hot_encoded = [0] * len(actions)

    if action in actions:
        index = actions.index(action)
        one_hot_encoded[index] = 1

    return one_hot_encoded

def encode_player_position(position):
    positions = ['Button', 'Small Blind', 'Big Blind', 'EP', 'MP', 'LP']
    one_hot_encoded = [0] * len(positions)

    if position in positions:
        index = positions.index(position)
        one_hot_encoded[index] = 1

    return one_hot_encoded

def deal_cards(num_players):
    # Define the deck of cards
    suits = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
    ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
    deck = [rank + suit for suit in suits for rank in ranks]

    # Shuffle the deck
    random.shuffle(deck)

    # Deal community cards
    community_cards = deck[:5]

    # Deal hole cards to players
    players_hole_cards = [deck[5 + i * 2: 7 + i * 2] for i in range(num_players)]

    return community_cards, players_hole_cards

# Usage example:
num_players = 9
community_cards, players_hole_cards = deal_cards(num_players)



# Evaluate the hands
def evaluate_hand(hand):
    # Mapping for face cards
    rank_mapping = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    # Convert ranks, handling face cards
    ranks = sorted([rank_mapping[card[:-1]] for card in hand])
    suits = [card[-1] for card in hand]


    # Check for flush
    is_flush = len(set(suits)) == 1

    # Check for straight, including A-2-3-4-5
    is_straight = ranks == list(range(min(ranks), min(ranks) + 5)) or ranks == [2, 3, 4, 5, 14]

    # Royal Flush
    if is_flush and ranks == [10, 11, 12, 13, 14]:
        return 10
    # Straight Flush
    elif is_flush and is_straight:
        return 9
    # Four of a Kind
    elif max(ranks.count(rank) for rank in ranks) == 4:
        return 8
    # Full House
    elif sorted(ranks.count(rank) for rank in ranks) == [2, 3]:
        return 7
    # Flush
    elif is_flush:
        return 6
    # Straight
    elif is_straight:
        return 5
    # Three of a Kind
    elif 3 in [ranks.count(rank) for rank in ranks]:
        return 4
    # Two Pair
    pair_counts = [ranks.count(rank) for rank in set(ranks)]
    if pair_counts.count(2) == 2:  # Check if there are exactly two pairs
        return 3

    # One Pair
    elif 2 in pair_counts:
        return 2

    # High Card
    else:
        return 1

def best_hand(cards):
    # Generate all combinations of five cards
    five_card_combinations = itertools.combinations(cards, 5)

    # Evaluate each combination and keep track of the best hand
    best_hand_value = 0
    best_hand = None
    for hand in five_card_combinations:
        hand_value = evaluate_hand(list(hand))
        if hand_value > best_hand_value:
            best_hand = hand
            best_hand_value = hand_value

    return best_hand_value, best_hand

# # Determine the winner
def determine_winner(players, community_cards):
    max_score = 0
    max_score_hands = []

    for player in players:
        score, hand = best_hand(player["holeCards"] + community_cards)
        if score > max_score:
            max_score = score
            max_score_hands = [(player, hand)]
        elif score == max_score:
            max_score_hands.append((player, hand))

    if len(max_score_hands) == 1:
        return players.index(max_score_hands[0][0]) 
    else:
        winner = compare_hands(max_score_hands)
        return players.index(winner) 

def compare_hands(max_score_hands):
    rank_mapping = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    # First, transform the hands into lists of ranks for easier comparison
    transformed_hands = [
        ([rank_mapping[card[:-1]] for card in hand], player) 
        for player, hand in max_score_hands
    ]

    # Sort the transformed hands. The sorting is based on the entire hand as a tuple,
    # which allows for a comprehensive comparison of all cards in the hand.
    transformed_hands.sort(key=lambda x: x[0], reverse=True)

    # The first element in sorted hands will be the winner
    return transformed_hands[0][1]

# Example usage
# # Print the winner
# print("The winner is Player", determine_winner(players))



#use LSTM2
def convert_round_to_tensor_LSTM(players, community_cards, pot, roundIn, current_player, totalWealth=9000 ):
    holder = []
    for seat in range(0, 9):
        zeros = [0] * 14
        if players[seat]['player'].get_name() == current_player['player'].get_name():
            holder += zeros
            continue
        if players[seat]['status'] == 'out' and players[seat]['action'] == 'none':
            holder += zeros
            continue
        action = players[seat]['action']
        if action == 'c':
            if roundIn == 0:
                action = 'check'
            else:
                action = 'call'
        if action == 'raise':
            if roundIn == 0:
                action = 'bet'
            else:
                action = 'raise'
        #5 actions
        holder += encode_player_action_lstm(action)
        #for now total wealth is just 1000 * 9, though later money / totalWealth
        holder += [players[seat]['player'].get_money() / totalWealth]
        holder += [players[seat]['amountIn'] / players[seat]['player'].get_money()]
        #6 positions
        holder += encode_player_position(players[seat]['position'])
        active = [1] if players[seat]['status'] == 'active' else [0]
        holder += active
        #total catagories = 14
    position = encode_player_position(current_player['position'])
    holder += position
    chips = current_player['player'].get_money() / totalWealth
    holder += [chips]
    amountIn = current_player['amountIn'] / current_player['player'].get_money()
    holder += [amountIn]
    cards = card_to_encoder(current_player['holeCards'])
    holder += cards
    cards = card_to_encoder(community_cards)
    holder += cards
    pot = pot / totalWealth
    holder += [pot]
    tensor = torch.tensor(holder, dtype=torch.float)
    return tensor.unsqueeze(0)

#Encode State for DQN
def convert_round_to_tesnor_DQN(players, community_cards, pot, roundIn, current_player, LSTM_output, totalWealth=9000):
    holder = []
    curName = current_player['player'].get_name()
    for seat in range(0, 10):
        zeros = [0] * 12
        if str(seat) == curName:
            continue
        for i in range(0, 9):
            if players[i]['player'].get_name() == str(seat):
                if players[i]['status'] == 'out' and players[i]['action'] == 'none':
                    holder += zeros
                    continue
                # 3 actions
                holder += encode_player_action_rl(players[i]['action'])
                holder += [players[i]['player'].get_money() / totalWealth]
                holder += [players[i]['amountIn'] / players[i]['player'].get_money()]
                # 6 positions
                holder += encode_player_position(players[i]['position'])
                active = [1] if players[i]['status'] == 'active' else [0]
                holder += active
                break
    position = encode_player_position(current_player['position'])
    holder += position
    chips = current_player['player'].get_money() / totalWealth
    holder += [chips]
    amountIn = current_player['amountIn'] / current_player['player'].get_money()
    holder += [amountIn]
    #maybe add roundIn?
    cards = card_to_encoder(current_player['holeCards'])
    holder += cards
    cards = card_to_encoder(community_cards)
    holder += cards
    pot = pot / totalWealth
    holder += [pot]
    #change this up maybe
    buyIn = (roundIn - current_player['roundIn']) / current_player['player'].get_money()
    holder += [buyIn]
    holder += LSTM_output
    tensor = torch.tensor(holder, dtype=torch.float)
    return tensor.unsqueeze(0)

def calculate_reward(player, state, action, next_state, done ):
    pass