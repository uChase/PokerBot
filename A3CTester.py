import torch
import DQN1
import pokergame
import pokerutils
import A3C
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


model = LSTM(239, 256, 2, 5).to(device)
model.load_state_dict(
    torch.load("./models/LSTM2.pth", map_location=torch.device(device))
)
input_dim = 210
output_dim = 3
num_layers = 2
gamma = 0.99

global_model_class = A3C.GlobalModel(input_dim, output_dim, num_layers)
global_model_class.load_model("./models/A3C.pth")

playerList = [pokergame.Player(str(i + 1), 1000) for i in range(9)]
totalWealth = 9000
games = 25000
numGameRounds = 0
numTimesWonMoney = 0
totalMoneyChange = 0
winnerArray = [0 for _ in range(len(playerList))]
for gameCount in range(games):
    if gameCount % 50 == 0:
        print(gameCount)
        numGameRounds += 1
        topMoney = 0
        bestPlayer = 0
        for i, player in enumerate(playerList):
            if player.get_money() > topMoney:
                topMoney = player.get_money()
                bestPlayerIndex = i
        winnerArray[bestPlayerIndex] += 1
        if playerList[4].get_money() > 1000:
            numTimesWonMoney += 1
        totalMoneyChange += playerList[4].get_money() - 1000
        for player in playerList:
            player.set_money(1000)
    game = pokergame.PokerGame()
    for player in playerList:
        game.add_player(player)
    enough = game.check_enough()
    if not enough:
        continue
    gameOver = False
    currplayer, cardStatus, community_cards, pot, roundIn = game.startGame()
    viewable_cards = []
    while not gameOver:
        # print("round")
        (
            gameStatus,
            currplayer,
            players,
            pot,
            cardStatus,
            roundIn,
        ) = game.check_round_redo()
        if gameStatus == "end":
            gameOver = True
            # loop through and get Q value of each player
            continue
        if currplayer["status"] == "out":
            gameStatus, currplayer, players, pot, cardStatus, roundIn = game.next_turn(
                "null"
            )
            continue
        if gameStatus == "redo":
            if cardStatus == 1:
                viewable_cards = community_cards[:3]
            elif cardStatus == 2:
                viewable_cards = community_cards[:4]
            elif cardStatus == 3:
                viewable_cards = community_cards[:5]
        lstmInput = pokerutils.convert_round_to_tensor_LSTM(
                game.players, viewable_cards, pot, roundIn, currplayer, game.totalWealth
            )
        lstmOut = []
        lstmIndex = 0
        move = None
        if currplayer["player"].get_name() != "5":
                with torch.no_grad():
                    hn = currplayer["player"].get_hn()
                    cn = currplayer["player"].get_cn()
                    lstmInput = lstmInput.unsqueeze(0).to(device)
                    round_output, (hn, cn) = model(
                        lstmInput,
                        (
                            hn.detach() if hn is not None else None,
                            cn.detach() if cn is not None else None,
                        ),
                    )
                    currplayer["player"].set_hn(hn)
                    currplayer["player"].set_cn(cn)
                    softmax = nn.Softmax(dim=1)
                    softmax_output = softmax(round_output)
                    lstmOut = softmax_output.squeeze().tolist()
                    lstmIndex = torch.argmax(softmax_output).item()
                if lstmIndex == 0:
                    move = "fold"
                elif lstmIndex == 1 or lstmIndex == 2:
                    move = "c"
                elif lstmIndex == 3 or lstmIndex == 4:
                    move = "raise"
        else:
            ahn = currplayer["player"].get_hn()
            acn = currplayer["player"].get_cn()
            chn = currplayer["player"].get_hn2()
            ccn = currplayer["player"].get_cn2()
            stateTensor = pokerutils.convert_round_to_tesnor_A3C(
                game.players,
                viewable_cards,
                pot,
                roundIn,
                currplayer,
                game.totalWealth,
            )
            stateTensor = stateTensor.to(device)
            action, policy, (ahn, acn), (chn, ccn) = global_model_class.choose_action(
                stateTensor, ahn, acn, chn, ccn
            )
            currplayer["player"].set_hn(ahn)
            currplayer["player"].set_cn(acn)
            currplayer["player"].set_hn2(chn)
            currplayer["player"].set_cn2(ccn)
            if action == 0:
                move = "fold"
            elif action == 1:
                move = "c"
            elif action == 2:
                move = "raise"
            else:
                print("ERROR")
        gameStatus, currplayer, players, pot, cardStatus, roundIn = game.next_turn(move)
    for player in playerList:
        player.reset_cn()
        player.reset_hn()
print(winnerArray)
print("num times won money", numTimesWonMoney)
print("num  rounds", numGameRounds)
print("total money change", totalMoneyChange)