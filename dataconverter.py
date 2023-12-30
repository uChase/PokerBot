import io
import os
from itertools import product
import re
import json

import copy

import torch

N_CARDS = 52

#try later a 2d tensor for cards, might do better in training 
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def card_to_encoder(cards):
    # Define the suits and ranks
    suits = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
    ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']

    # Create a mapping from card to index with ranks grouped together
    card_to_index = {f"{rank}{suit}": index for index, (rank, suit) in enumerate(product(ranks, suits))}

    # Initialize a 52-length binary array
    encoder = [0] * 52

    # Set the corresponding indices to 1 for each card
    for card in cards:
        encoder[card_to_index[card]] = 1

    return encoder


#get rid of bets, keep only raises
def encode_player_action(action):
    # actions = ['folds', 'checks', 'calls', 'bets', 'raises']
    if action == 'bets':
        action = 'raises'
    actions = ['folds', 'checks', 'calls', 'raises']
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


def getSeatData(line):
    pattern = re.compile(r'Seat (\d+): (.*?) \(([\d.]+)\)')
    match = pattern.match(line)
    if match:
        seat_number, player_name, money = match.groups()
        return int(seat_number), player_name.strip(), float(money)
    return None

def assign_table_positions(players_data):
    # Find the button position
    button_seat = None
    # print(players_data)
    for seat, data in players_data.items():
        if data['pos'] == 'button':
            button_seat = seat
            break

    if button_seat is None:
        raise ValueError("No button position found")

    positions = ["Button", "Small Blind", "Big Blind", "EP", "MP"]
    seat_positions = {}
    seats = list(players_data.keys())
    total_seats = len(seats)
    button_index = seats.index(button_seat)

    for i in range(total_seats):
        current_index = (button_index + i) % total_seats
        current_seat = seats[current_index]
        if i < len(positions):
            position = positions[i]
        elif i == total_seats - 1:
            position = "LP"
        else:
            position = "MP"  # Assign MP to the remaining seats except the last one
        seat_positions[current_seat] = {'name': players_data[current_seat]['name'], 'stack': players_data[current_seat]['stack'], 'action': players_data[current_seat]['action'], 'pos': position, 'cards': players_data[current_seat]['cards'], 'amountIn': players_data[current_seat]['amountIn'], "active": players_data[current_seat]['active'], "currentPutIn": players_data[current_seat]['currentPutIn']}

    return seat_positions

def reset_player_actions(players):
    # Clone the player data and reset their actions
    cloned_players = copy.deepcopy(players)
    for player in cloned_players.values():
        player['action'] = 'none'
    return cloned_players

def load_data(file_path):
    def getGameSegment(start_line=0,  previous_players=None, prev_pot=0, prev_total_wealth = 0, prev_state={'pot': 0, 'flop': [], 'turn': '', 'river': ''} ):
        with open(file_path, 'r') as file:
            players = previous_players if previous_players else {}
            gameState = prev_state
            totalWealth = prev_total_wealth
            pot = prev_pot
            button_seat = None
            blinds_pattern = re.compile(r'Player (\w+) has (small|big) blind \(([\d.]+)\)')
            bet_pattern = re.compile(r'Player (\w+) (raises|calls|bets) \(([\d.]+)\)')
            card_pattern = re.compile(r'Player (\w+) received card: \[([\w\d]+)\]')
            safe_pattern = re.compile(r'Player (\w+) (folds|checks)')
            flop_pattern = re.compile(r'\*\*\* FLOP \*\*\*: \[([\w\d\s]+)\]')
            turn_pattern = re.compile(r'\*\*\* TURN \*\*\*: \[[\w\d\s]+\] \[([\w\d]+)\]')
            river_pattern = re.compile(r'\*\*\* RIVER \*\*\*: \[[\w\d\s]+\] \[([\w\d]+)\]')

            for i, line in enumerate(file):
                if i < start_line:
                    continue
                if "Game ended" in line or "------ Summary -----" in line:
                    return None, i

                if line.startswith('Seat'):
                    # print(line)
                    seat_data = getSeatData(line)
                    # print(seat_data)
                    if seat_data:
                        seat_number, player_name, money = seat_data
                        if seat_number == button_seat:
                            players[seat_number] = {'name': player_name, 'stack': money, 'pos': "button", "action": "none", "cards": [], "amountIn": 0, "active": True, "currentPutIn": 0}
                        else:
                            players[seat_number] = {'name': player_name, 'stack': money, 'pos': "no", "action": "none", "cards": [], "amountIn": 0, "active": True, "currentPutIn": 0}                        
                        totalWealth += money
                    elif 'is the button' in line:
                        button_seat = int(line.split()[1])  # Extract the seat number of the button
                elif blinds_match := blinds_pattern.match(line):
                    player_name, blind_type, amount = blinds_match.groups()
                    amount = float(amount)
                    pot += amount
                    for player in players.values():
                        if player['name'] == player_name:
                            player['stack'] -= amount
                            player['amountIn'] += amount
                            player['action'] = f'{blind_type} blind'
                            break
                elif action_match := bet_pattern.match(line):
                    player_name, action, amount = action_match.groups()
                    amount = float(amount)
                    for player in players.values():
                        if player['name'] == player_name:
                            player['action'] = action
                            if action in ['raises', 'calls', 'bets']:
                                player['stack'] -= amount
                                pot += amount
                                player['amountIn'] += amount
                                player['currentPutIn'] = amount
                            if player_name == 'IlxxxlI':
                                if previous_players == None:
                                    players = assign_table_positions(players)
                                gameState['pot'] = pot
                                return (players, gameState, totalWealth), i + 1
                elif safe_match := safe_pattern.match(line):
                     player_name, action = safe_match.groups()
                     for player in players.values():
                        if player['name'] == player_name:
                            player['action'] = action
                            if action == 'folds':
                                player['active'] = False
                                # play with this
                                # totalWealth -= player['stack']
                            if player_name == 'IlxxxlI':
                                if previous_players == None:
                                    players = assign_table_positions(players)
                                player['currentPutIn'] = 0
                                gameState['pot'] = pot
                                return (players, gameState, totalWealth), i + 1
                elif card_match := card_pattern.match(line):
                    player_name, card = card_match.groups()
                    if player_name == 'IlxxxlI':
                        for player in players.values():
                            if player['name'] == player_name:
                                player['cards'].append(card)
                                continue
                flop_match = flop_pattern.match(line)
                if flop_match:
                    gameState['flop'] = flop_match.group(1).split()

                turn_match = turn_pattern.match(line)
                if turn_match:
                    gameState['turn'] = turn_match.group(1)

                river_match = river_pattern.match(line)
                if river_match:
                    gameState['river'] = river_match.group(1)
            if previous_players == None:
                players = assign_table_positions(players)
            gameState['pot'] = pot
            return (players, gameState, totalWealth), i + 1

    # Collect multiple game segments
    all_games = []
    game_tensors = []
    with open(file_path, 'r') as file:
        game_segments = []
        start_line = 0
        lineCount = 0
        while True:
            line = file.readline()
            if not line:
                break
            if "Game started at" in line: 
                    accumulated_tensor = torch.empty(0, 244)
                    start_line = lineCount
                    prev_pot = 0
                    prev_state = {'pot': 0, 'flop': [], 'turn': '', 'river': ''}
                    prev_total_wealth = 0
                    previous_players = None
                    while True:
                        segment, next_line = getGameSegment(start_line, previous_players, prev_pot, prev_total_wealth, prev_state)
                        if segment is None:
                            break
                        game_segments.append(segment)
                        previous_players, state, prev_total_wealth = segment
                        roundTensor = convert_round_to_tensor(previous_players, state, prev_total_wealth)
                        if accumulated_tensor.nelement() == 0:
                            accumulated_tensor = roundTensor
                        else:
                            accumulated_tensor = torch.cat((accumulated_tensor, roundTensor), dim=0)
                        # print(segment)
                        previous_players = reset_player_actions(previous_players)
                        start_line = next_line
                        prev_pot = state['pot']
                        prev_state = state.copy()

                    # all_games.append(game_segments)
                    game_tensors.append(accumulated_tensor)
            lineCount += 1
            # print(lineCount)

    output_json_path = file_path.rsplit('.txt', 1)[0] + '.json'
    return game_tensors
    # with open(output_json_path, 'w') as json_file:
    #     json.dump(all_games, json_file, indent=4)

def convert_round_to_tensor(players, gameState, totalWealth):
    holder = []
    # print(players)
    for seat in range(1, 10):
        seat_key = int(seat)
        zeros = [0] * 13
        if seat_key in players:

            if players[seat_key]['name'] == 'IlxxxlI':
                ### add zero vector
                holder += zeros
                continue
            if players[seat_key]['active'] == False and players[seat_key]['action'] == 'none':
                ### folded status only
                holder += zeros
                continue
            #4 actions
            actions = encode_player_action(players[seat_key]['action'])
            # print("actions ")
            # print(actions)
            holder += actions
            # print("chips")
            chips = players[seat_key]['stack'] / totalWealth
            # print(chips)
            holder += [chips]
            # print("amount in")
            amountIn = players[seat_key]['amountIn'] / players[seat_key]['stack']
            # print(amountIn)
            holder += [amountIn]
            # print("pos")
            #6 positions
            position = encode_player_position(players[seat_key]["pos"])
            holder += position
            # print(position)
            # print("active")
            active = [1] if players[seat_key]['active'] else [0]
            holder += active
            # print(active)
            #13 per player
            #117 total
 
        else:
            holder += zeros
            # print("seat is not in ", seat)
            #zeros
    aiAction = []
    for seat in range(1, 10):
        seat_key = int(seat)
        if seat_key in players:
            if players[seat_key]['name'] == 'IlxxxlI':
                #5 slots
                # print("player pos")
                position = encode_player_position(players[seat_key]["pos"])
                holder += position
                # print(position)
                # print("player chips")
                chips = (players[seat_key]['stack'] + players[seat_key]['currentPutIn']) / totalWealth
                holder += [chips]
                # print(chips)
                # print("player amount in")
                amountIn = (players[seat_key]['amountIn'] - players[seat_key]['currentPutIn']) / (players[seat_key]['stack'] + players[seat_key]['currentPutIn'])
                holder += [amountIn]
                # print(amountIn)
                # print("player cards")
                cards = card_to_encoder(players[seat_key]['cards'])
                holder += cards
                # print(cards)
                # print("player action")
                aiAction = encode_player_action(players[seat_key]['action'])
                # print(aiAction)
                break
    cards = []
    if len(gameState['flop']) == 3:
        cards = gameState['flop']
        if gameState['turn'] != '':
            cards += [gameState['turn']]
            if gameState['river'] != '':
                cards += [gameState['river']]
    holder += card_to_encoder(cards)
    pot = gameState['pot'] / totalWealth
    holder += [pot]
    holder += aiAction
    print(len(holder))
    tensor = torch.tensor(holder, dtype=torch.float)
    return tensor.unsqueeze(0)

# # Example usage
games = load_data("./data/IlxxxlI/testing3.txt")
# players = assign_table_positions(players)
torch.save(games, 't3.pt')
