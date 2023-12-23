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

def card_to_encoder(card):
    # Define the suits and ranks
    suits = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
    ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']

    # Create a mapping from card to index with ranks grouped together
    card_to_index = {f"{rank}{suit}": index for index, (rank, suit) in enumerate(product(ranks, suits))}

    # Initialize a 52-length binary array with uint8 data type
    encoder = np.zeros(52, dtype=np.uint8)

    # Set the corresponding index to 1
    encoder[card_to_index[card]] = 1

    return encoder

def getSeatData(line):
    pattern = re.compile(r'Seat (\d+): ([^\(]+) \(([\d.]+)\)')
    match = pattern.match(line)
    if match:
        seat_number, player_name, money = match.groups()
        return int(seat_number), player_name.strip(), float(money)
    return None

def assign_table_positions(players_data):
    # Find the button position
    button_seat = None
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
        seat_positions[current_seat] = {'name': players_data[current_seat]['name'], 'stack': players_data[current_seat]['stack'], 'action': players_data[current_seat]['action'], 'pos': position, 'cards': players_data[current_seat]['cards'], 'amountIn': players_data[current_seat]['amountIn']}

    return seat_positions

def reset_player_actions(players):
    # Clone the player data and reset their actions
    cloned_players = copy.deepcopy(players)
    for player in cloned_players.values():
        player['action'] = 'none'
    return cloned_players

def load_data(file_path):
    def getGameSegment(start_line=0,  previous_players=None, prev_pot=0, prev_total_wealth = 0, prev_state={'pot': 0} ):
        with open(file_path, 'r') as file:
            players = previous_players if previous_players else {}
            gameState = prev_state
            totalWealth = prev_total_wealth
            pot = prev_pot
            button_seat = None
            blinds_pattern = re.compile(r'Player (\w+) has (small|big) blind \(([\d.]+)\)')
            bet_pattern = re.compile(r'Player (\w+) (raises|calls) \(([\d.]+)\)')
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
                    seat_data = getSeatData(line)
                    if seat_data:
                        seat_number, player_name, money = seat_data
                        if seat_number == button_seat:
                            players[seat_number] = {'name': player_name, 'stack': money, 'pos': "button", "action": "none", "cards": [], "amountIn": 0}
                        else:
                            players[seat_number] = {'name': player_name, 'stack': money, 'pos': "no", "action": "none", "cards": [], "amountIn": 0}                        
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
                            if action in ['raises', 'calls']:
                                player['stack'] -= amount
                                pot += amount
                                player['amountIn'] += amount

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
                            if player_name == 'IlxxxlI':
                                if previous_players == None:
                                    players = assign_table_positions(players)
                                gameState['pot'] = pot
                                return (players, gameState, totalWealth), i + 1
                elif card_match := card_pattern.match(line):
                    player_name, card = card_match.groups()
                    if player_name == 'IlxxxlI':
                        for player in players.values():
                            if player['name'] == player_name:
                                player['cards'].append(card)
                                break
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
    with open(file_path, 'r') as file:
        game_segments = []
        start_line = 0
        lineCount = 0
        while True:
            line = file.readline()
            if not line:
                break
            if "Game started at" in line: 
                    game_segments = []
                    start_line = lineCount
                    prev_pot = 0
                    prev_state = {'pot': 0}
                    prev_total_wealth = 0
                    previous_players = None
                    while True:
                        segment, next_line = getGameSegment(start_line, previous_players, prev_pot, prev_total_wealth, prev_state)
                        if segment is None:
                            break
                        game_segments.append(segment)
                        previous_players, state, prev_total_wealth = segment
                        previous_players = reset_player_actions(previous_players)
                        start_line = next_line
                        prev_pot = state['pot']
                        prev_state = state
                    all_games.append(game_segments)
            lineCount += 1
    output_json_path = file_path.rsplit('.txt', 1)[0] + '.json'

    with open(output_json_path, 'w') as json_file:
        json.dump(all_games, json_file, indent=4)
                

# Example usage
games = load_data("./data/IlxxxlI/d3.txt")
# players = assign_table_positions(players)
