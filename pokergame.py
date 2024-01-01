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
        self.totalWealth = 0

    def add_player(self, player):
        if player.get_money() <= 0:
            return
        self.players.append({"player": player, "amountIn": 0, "status": "active", "action": "None", "holeCards": [], "position": "None", "roundIn": 0, "Q": 0, "roundLoop": 0})

    def assign_table_positions(self):
        random.shuffle(self.players)
        positions = ["Button", "Small Blind", "Big Blind", "EP", "EP", "MP", "MP", "MP", "LP"]
        for i, player in enumerate(self.players):
            self.players[i]["position"] = positions[i]

    def check_enough(self):
        if len(self.players) < 4:
            return False
        return True

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
            self.totalWealth = self.totalWealth + player["player"].get_money()
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

        if self.current_player["player"].get_money() <= 0:
            # print("out")
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
            if self.current_player["player"].get_money() - self.roundIn <= 0:
                self.current_player["status"] = "out"
                self.current_player["action"] = "none"
                self.current_player['player'].set_money(0)
                index = self.players.index(self.current_player)
                if index == len(self.players) - 1:
                    self.current_player = self.players[0]
                else:
                    self.current_player = self.players[index + 1]
                return "cont", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn
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
            if self.current_player["player"].get_money() - (self.roundIn + 15) <= 0:
                self.current_player["status"] = "out"
                self.current_player["action"] = "none"
                self.current_player['player'].set_money(0)
                index = self.players.index(self.current_player)
                if index == len(self.players) - 1:
                    self.current_player = self.players[0]
                else:
                    self.current_player = self.players[index + 1]
                return "cont", self.current_player, self.players, self.pot, self.cardStatus, self.roundIn

            self.current_player["action"] = "raise"
            self.roundLoop += 1
            self.current_player["roundLoop"] = self.roundLoop


            self.current_player["amountIn"] += self.roundIn - self.current_player["roundIn"] + 15
            self.current_player["player"].pay_amount(self.roundIn - self.current_player["roundIn"] + 15)
            self.pot += self.roundIn - self.current_player["roundIn"] + 15
            self.current_player["roundIn"] = self.roundIn + 15
            self.roundIn += 15
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
            #WINNER IF ALL FOLD
            inPlayers[0]["Q"] = 0.75 * self.pot / self.totalWealth * inPlayers[0]['player'].get_money() / self.totalWealth
            inPlayers[0]["result"] = "solo win"
            inPlayers[0]["player"].receive_amount(self.pot)
            for player in self.players:
                if player["status"] == "out":
                    if player['player'].get_money() <= 0:
                        player["Q"] = -10
                        print("bankrupt")
                        player["result"] = "bankrupt"
                        continue
                    clonePlayers = inPlayers.copy()
                    clonePlayers.append(player)
                    winnerIndex = pokerutils.determine_winner(clonePlayers, self.community_cards)
                    if winnerIndex == len(clonePlayers) - 1:
                        #would have won if hadnt folded
                        # player["Q"] = -0.02 * self.pot / self.totalWealth * (1 - player['player'].get_money() / self.totalWealth)
                        player["Q"] =   (1 - player["amountIn"] / self.pot) * player['player'].get_money() / self.totalWealth * -.01
                        player["result"] = "bad fold"
                    else:
                        #would have lost if hadnt folded
                        player["Q"] = (1 - player["amountIn"] / self.pot) * player['player'].get_money() / self.totalWealth * .01
                        player["result"] = "good fold"
        else: 
            winnerIndex = pokerutils.determine_winner(inPlayers, self.community_cards)
            #WON SHOWDOWN
            inPlayers[winnerIndex]["Q"] =  self.pot / self.totalWealth * inPlayers[winnerIndex]['player'].get_money() / self.totalWealth
            inPlayers[winnerIndex]["result"] = "win"
            inPlayers[winnerIndex]["player"].receive_amount(self.pot)
            winnerName = inPlayers[winnerIndex]["player"].get_name()
            for player in self.players:
                if player["status"] == "out":
                    if player['player'].get_money() <= 0:
                        player["Q"] = -10
                        print("bankrupt")
                        player["result"] = "bankrupt"
                        continue
                    clonePlayers = inPlayers.copy()
                    clonePlayers.append(player)
                    winnerIndex = pokerutils.determine_winner(clonePlayers, self.community_cards)
                    if winnerIndex == len(clonePlayers) - 1:
                        # player["Q"] = -0.02 * self.pot / self.totalWealth * (1 - player['player'].get_money() / self.totalWealth)
                        player["Q"] =   (1 - player["amountIn"] / self.pot) * player['player'].get_money() / self.totalWealth * -.01
                        player["result"] = "bad fold"
                    else:
                        player["Q"] =   (1 - player["amountIn"] / self.pot) * player['player'].get_money() / self.totalWealth * .01
                        player["result"] = "good fold"
                elif player["player"].get_name() != winnerName:
                    player["Q"] = -1 * player["amountIn"] / self.totalWealth * (1 - player['player'].get_money() / self.totalWealth)
                    player["result"] = "loss"
        


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