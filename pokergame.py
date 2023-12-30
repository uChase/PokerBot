import pokerutils
import random

class PokerGame:
    def __init__(self):
        self.players = []
        self.current_player = None
        self.pot = 0
        self.community_cards = []
        self.cardStatus = 0
        self.roundIn = 0
        self.roundLoop = 1

    def add_player(self, player):
        self.players.append({"player": player, "amountIn": 0, "status": "active", "action": "None", "holeCards": [], "position": "None", "roundIn": 0, "Q": 0, "roundLoop": 0})

    def assign_table_positions(self):
        random.shuffle(self.players)
        positions = ["Button", "Small Blind", "Big Blind", "EP", "EP", "MP", "MP", "MP", "LP"]
        for i, player in enumerate(self.players):
            self.players[i]["position"] = positions[i]

    def startGame(self):
        self.init_round()
        return self.current_player,  self.cardStatus, self.community_cards, self.pot, self.roundIn



    def init_round(self):
        self.assign_table_positions()
        community_cards, players_hole_cards = pokerutils.deal_cards(len(self.players))
        self.players[1]['player'].pay_amount(1)
        self.players[1]["amountIn"] = 1
        self.players[1]["roundIn"] = 1
        self.players[2]['player'].pay_amount(2)
        self.players[2]["amountIn"] = 2
        self.players[2]["roundIn"] = 2
        self.players[2]["roundLoop"] = 1
        self.pot += 3
        self.roundIn = 2
        self.current_player = self.players[3]
        self.community_cards = community_cards
        for i, player in enumerate(self.players):
            self.players[i]["holeCards"] = players_hole_cards[i]
        pass


    def check_round_redo(self):
        numActive = 0
        for player in self.players:
            if player["status"] == "active":
                numActive += 1
        if numActive == 1:
            self.end_game()
            return "end", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn
        

        if self.current_player["roundLoop"] == self.roundLoop:
            if self.cardStatus == 3:
                self.end_game()
                return "end", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn
            self.cardStatus += 1
            self.roundLoop = 1
            self.roundIn = 0
            for player in self.players:
                player["roundLoop"] = 0
                player["roundIn"] = 0
            return "redo", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn
        return "cont", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn

    def next_turn(self, move):

        if self.current_player["player"].get_money() == 0:
            print("out")
            self.current_player["status"] = "out"
            self.current_player["action"] = "none"
            index = self.players.index(self.current_player)
            if index == len(self.players) - 1:
                self.current_player = self.players[0]
            else:
                self.current_player = self.players[index + 1]
            return "cont", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn

        if self.current_player["roundLoop"] == self.roundLoop:
            if self.cardStatus == 3:
                self.end_game()
                return "end", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn
            self.cardStatus += 1
            self.roundLoop = 1
            self.roundIn = 0
            for player in self.players:
                player["roundLoop"] = 0
                player["roundIn"] = 0
            return "redo", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn

        if self.current_player["status"] == "out":
            index = self.players.index(self.current_player)
            self.current_player["action"] = "none"
            self.current_player["roundLoop"] = self.roundLoop
            if index == len(self.players) - 1:
                self.current_player = self.players[0]
            else:
                self.current_player = self.players[index + 1]
            return "cont", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn
        

        if move == "fold":
            self.current_player["action"] = "fold"
            self.current_player["status"] = "out"
            self.current_player["roundLoop"] = self.roundLoop
            index = self.players.index(self.current_player)
            if index == len(self.players) - 1:
                self.current_player = self.players[0]
            else:
                self.current_player = self.players[index + 1]
        elif move == "c":
            self.current_player["action"] = "c"
            self.current_player["roundLoop"] = self.roundLoop
            self.current_player["amountIn"] += self.roundIn - self.current_player["roundIn"]
            self.pot += self.roundIn - self.current_player["roundIn"]
            self.current_player["player"].pay_amount(self.roundIn - self.current_player["roundIn"])
            self.current_player["roundIn"] = self.roundIn
            index = self.players.index(self.current_player)
            if index == len(self.players) - 1:
                self.current_player = self.players[0]
            else:
                self.current_player = self.players[index + 1]
        
        elif move == "raise":
            self.current_player["action"] = "raise"
            self.roundLoop += 1
            self.current_player["roundLoop"] = self.roundLoop
            self.current_player["amountIn"] += self.roundIn - self.current_player["roundIn"] + .01 * self.current_player["player"].get_money()
            self.pot += self.roundIn - self.current_player["roundIn"] + .01 * self.current_player["player"].get_money()
            self.current_player["roundIn"] = self.roundIn + .01 * self.current_player["player"].get_money()
            self.roundIn += .01 * self.current_player["player"].get_money()
            self.current_player["player"].pay_amount(self.roundIn - self.current_player["roundIn"] + .01 * self.current_player["player"].get_money())
            index = self.players.index(self.current_player)
            if index == len(self.players) - 1:
                self.current_player = self.players[0]
            else:
                self.current_player = self.players[index + 1]


        return "cont", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn

    def end_game(self):
        inPlayers = []
        for player in self.players:
            if player["status"] == "active":
                inPlayers.append(player)
        if len(inPlayers) == 1:
            inPlayers[0]["Q"] = self.pot / player["player"].get_money() * 500
            inPlayers[0]["player"].receive_amount(self.pot)
            for player in self.players:
                if player["status"] == "out":
                    clonePlayers = inPlayers.copy()
                    clonePlayers.append(player)
                    winnerIndex = pokerutils.determine_winner(clonePlayers, self.community_cards)
                    if winnerIndex == len(clonePlayers) - 1:
                        player["Q"] = self.pot /  player["player"].get_money()  * -100
                    else:
                        player["Q"] = clonePlayers[winnerIndex]["amountIn"] /  player["player"].get_money()  * 50
        else: 
            #TODO implement punishment for folding with good hand, and reward for folding with bad hand
            winnerIndex = pokerutils.determine_winner(inPlayers, self.community_cards)
            inPlayers[winnerIndex]["Q"] = self.pot / inPlayers[winnerIndex]["player"].get_money() * 500
            inPlayers[winnerIndex]["player"].receive_amount(self.pot)
            winnerName = inPlayers[winnerIndex]["player"].get_name()
            for player in self.players:
                if player["status"] == "out":
                    clonePlayers = inPlayers.copy()
                    clonePlayers.append(player)
                    winnerIndex = pokerutils.determine_winner(clonePlayers, self.community_cards)
                    if winnerIndex == len(clonePlayers) - 1:
                        player["Q"] = self.pot /  player["player"].get_money()  * -100
                    else:
                        player["Q"] = clonePlayers[winnerIndex]["amountIn"] /  player["player"].get_money()  * 50
                elif player["player"].get_name() != winnerName:
                    player["Q"] = player["amountIn"] /  player["player"].get_money()  * -100
        


class Player:
    def __init__(self, name, money):
        self.name = name
        self.money = money
        self.hn = None
        self.cn = None

    def get_money(self):
        return self.money
    
    def set_money(self, money):
        self.money = money
    
    def pay_amount(self, amount):
        self.money -= amount
    
    def receive_amount(self, amount):
        self.money += amount

    def get_name(self):
        return self.name
    
    def set_hn(self, hn):
        self.hn = hn
    
    def set_cn(self, cn):
        self.cn = cn
    
    def get_hn(self):
        return self.hn

    def get_cn(self):
        return self.cn