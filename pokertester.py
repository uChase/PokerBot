import pokergame
import pokerutils
import torch
import torch.nn as nn
import torch.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gameOver = False
p1 = pokergame.Player("1", 1000)
p2 = pokergame.Player("2", 1000)
p3 = pokergame.Player("3", 1000)
p4 = pokergame.Player("4", 1000)
p5 = pokergame.Player("5", 1000)
p6 = pokergame.Player("6", 1000)
p7 = pokergame.Player("7", 1000)
p8 = pokergame.Player("8", 1000)
p9 = pokergame.Player("9", 1000)
game = pokergame.PokerGame()
game.add_player(p1)
game.add_player(p2)
game.add_player(p3)
game.add_player(p4)
game.add_player(p5)
game.add_player(p6)
game.add_player(p7)
game.add_player(p8)
game.add_player(p9)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Using output of the last round

        return out, (hn, cn)


# experiment with this model 1 uses 128, 2 256
hidden_size = 256
# 244 features - 5 at the end which are the labels
input_size = 239
output_size = 5
num_layers = 2
learning_rate = 0.001

model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('./models/LSTM2.pth'))

currplayer, cardStatus, community_cards, pot, roundIn = game.startGame()

viewable_cards = []

while not gameOver:
    for player in game.players:
        print(player["player"].get_name(), player["player"].get_money())
    if currplayer["status"] == "out":
        gameStatus, currplayer, players, pot, cardStatus, roundIn = game.next_turn("null")
        continue

    lstmInput = pokerutils.convert_round_to_tensor_LSTM(game.players, viewable_cards, pot, roundIn, currplayer)

    lstmOut = []
    with torch.no_grad():
        hn = currplayer["player"].get_hn()
        cn = currplayer["player"].get_cn()
        lstmInput = lstmInput.unsqueeze(0).to(device)
        round_output, (hn, cn) = model(lstmInput, (hn.detach() if hn is not None else None,
                                                    cn.detach() if cn is not None else None))
        currplayer["player"].set_hn(hn)
        currplayer["player"].set_cn(cn)
        softmax = nn.Softmax(dim=1)
        softmax_output = softmax(round_output)
        lstmOut = softmax_output.squeeze().tolist()
        
            
            
    stateTensor = pokerutils.convert_round_to_tesnor_DQN(game.players, viewable_cards, pot, roundIn, currplayer, lstmOut)
    gameStatus, currplayer, players, pot, cardStatus, roundIn = game.check_round_redo()
    if gameStatus == "redo":
        print("redo")
        if cardStatus == 1:
            viewable_cards = community_cards[:3]
        elif cardStatus == 2:
            viewable_cards = community_cards[:4]
        elif cardStatus == 3:
            viewable_cards = community_cards[:5]
    if gameStatus == "end":
        print("Game Over!")
        print(players)
        print("community cards: ", community_cards)
        gameOver = True
        continue
    
    move = input("Enter your move, " + currplayer["player"].get_name() + ": ")
    gameStatus, currplayer, players, pot, cardStatus, roundIn = game.next_turn(move)

    
