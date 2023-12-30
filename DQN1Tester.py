import torch
import DQN1
import pokergame
import pokerutils
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#INIT LSTM
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
    
hidden_size = 256
input_size = 239
output_size = 5
num_layers = 2
learning_rate = 0.001
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('./models/LSTM2.pth'))

#INIT DQN
input_dim = 215
output_dim = 3

playerList = [pokergame.Player(str(i+1), 1000) for i in range(9)]
models = [DQN1.DQN(input_dim, output_dim).to(device) for _ in range(len(playerList))]
for i in range(len(models)):
    models[i].load_state_dict(torch.load('./models/DQN1_' + str(i) + '.pth'))
    models[i].eval()

games = 10
numraise = 0
for game in range(games):
    players = []
    end = False
    for player in playerList:
        if player.get_money() <=  50:
            end = True
    if end:
        break
    game = pokergame.PokerGame()
    for player in playerList:
        game.add_player(player)
    gameOver = False
    currplayer, cardStatus, community_cards, pot, roundIn = game.startGame()
    viewable_cards = []

    while not gameOver:
        gameStatus, currplayer, players, pot, cardStatus, roundIn = game.check_round_redo()
        if gameStatus == "end":            
            gameOver = True
            #loop through and get Q value of each player
            continue
        if currplayer["status"] == "out":
            gameStatus, currplayer, players, pot, cardStatus, roundIn = game.next_turn("null")
            continue
        if gameStatus == "redo":
            if cardStatus == 1:
                viewable_cards = community_cards[:3]
            elif cardStatus == 2:
                viewable_cards = community_cards[:4]
            elif cardStatus == 3:
                viewable_cards = community_cards[:5]
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
        stateTensor = stateTensor.to(device)
        q_values = models[int(currplayer["player"].get_name())-1](stateTensor).to(device)
        action = torch.argmax(q_values).item()
        max_index = action
        move = ""
        if max_index == 0:
            move = "fold"
        elif max_index == 1:
            move = "c"
        elif max_index == 2:
            numraise += 1
            move = "raise"
        gameStatus, currplayer, players, pot, cardStatus, roundIn = game.next_turn(move)
    print(players)
    print("community cards: ", community_cards)
print(numraise)