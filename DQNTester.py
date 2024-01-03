import torch
import DQN1
import pokergame
import pokerutils
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# INIT LSTM
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
model.load_state_dict(
    torch.load("./models/LSTM2.pth", map_location=torch.device(device))
)

# INIT DQN
input_dim = 215
output_dim = 3

playerList = [pokergame.Player(str(i + 1), 1000) for i in range(9)]
models = [DQN1.DQN(input_dim, output_dim).to(device) for _ in range(len(playerList))]
for i in range(len(models)):
    if i == 4:
        models[i].load_state_dict(
            torch.load("./models/DQN_V_LSTM.pth", map_location=torch.device(device))
        )
        models[i].eval()
    else:
        models[i].load_state_dict(
            torch.load("./models/DQN_4_2.pth", map_location=torch.device(device))
        )
        models[i].eval()


games = 1
numGameRounds = 0
numTimesWonMoney = 0
totalMoneyChange = 0
winnerArray = [0 for _ in range(len(playerList))]
for gameCount in range(games):
    players = []
    state_action_pairs = {p_id: [] for p_id in range(len(playerList))}

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

    end = False
    for player in playerList:
        if player.get_money() <= 50:
            end = True
    if end:
        continue
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
            game.players, viewable_cards, pot, roundIn, currplayer
        )
        lstmOut = []
        lstmIndex = 0
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
        stateTensor = pokerutils.convert_round_to_tesnor_DQN(
            game.players, viewable_cards, pot, roundIn, currplayer, lstmOut
        )
        stateTensor = stateTensor.to(device)
        q_values = models[int(currplayer["player"].get_name()) - 1](stateTensor).to(
            device
        )
        action = torch.argmax(q_values).item()
        state_action_pairs[int(currplayer["player"].get_name()) - 1].append(
            (stateTensor, action, currplayer["player"].get_money(), pot, roundIn)
        )
        max_index = action
        move = ""
        # test against raw lstm
        if currplayer["player"].get_name() == "5":
            if max_index == 0:
                move = "fold"
            elif max_index == 1:
                move = "c"
            elif max_index == 2:
                move = "raise"
        else:
            if lstmIndex == 0:
                move = "fold"
            elif lstmIndex == 1 or lstmIndex == 2:
                move = "c"
            elif lstmIndex == 3 or lstmIndex == 4:
                move = "raise"
        gameStatus, currplayer, players, pot, cardStatus, roundIn = game.next_turn(move)
    for player in game.players:
        print(
            player["player"].get_name(),
            player["player"].get_money(),
            player["Q"] * (10**6),
            player["status"],
            player["holeCards"],
        )
    print(community_cards)
    # for player in players:
    #     agent_id = int(player['player'].get_name()) - 1
    #     endReward = player['Q']
    #     endResult = player['result']
    #     print(agent_id + 1)
    #     for i in range(len(state_action_pairs[agent_id])):
    #         state, action, bankroll, pot, roundIn = state_action_pairs[agent_id][i]
    #         done = i == len(state_action_pairs[agent_id]) - 1
    #         next_state = torch.zeros_like(state) if done else state_action_pairs[agent_id][i + 1][0]
    #         reward = 0
    #         if done:
    #             reward = endReward
    #         else:
    #             reward = pokerutils.calculate_reward(bankroll, action, endResult, pot, roundIn, 9000)
    #         reward *= (10 ** 6)
    #         print(reward)
# for player in playerList:
#     print(player.get_name(), player.get_money())

# print(winnerArray)
# print("num times won money", numTimesWonMoney)
# print("num  rounds", numGameRounds)
# print("total money change", totalMoneyChange)
# Model DQN_4_2 and DQN_2_2 outpreforms all other models
