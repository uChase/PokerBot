import torch
import DQN1
import pokergame
import pokerutils
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


hidden_size = 256
# 244 features - 5 at the end which are the labels
input_size = 239
output_size = 5
num_layers = 2
learning_rate = 0.0001

model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('./models/LSTM2.pth'))


epochs = 100000
batch_size = 256
input_dim = 215
output_dim = 3
capacity = 100000
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.9999999

playerList = [pokergame.Player(str(i+1), 1000) for i in range(9)]
agent = DQN1.PokerAgent(input_dim, output_dim, capacity, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)

agent.loadModel('./models/DQN_4_2.pth')

total_wealth = 9000

for epoch in range(epochs):
    players = []
    if epoch % 50 == 0:
        print("Epoch: ", epoch)
    if epoch % 15 == 0:
        resetMoney = True
    if resetMoney:
        for player in playerList:
            player.set_money(1000)
    game = pokergame.PokerGame()
    for player in playerList:
        game.add_player(player)
    gameOver = False
    enough = game.check_enough()
    if not enough:
        continue
    currplayer, cardStatus, community_cards, pot, roundIn = game.startGame()
    viewable_cards = []
    state_action_pairs =  [] 
    while not gameOver:
        gameStatus, currplayer, players, pot, cardStatus, roundIn = game.check_round_redo()
        if gameStatus == "end":            
            gameOver = True
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
        lstmInput = pokerutils.convert_round_to_tensor_LSTM(game.players, viewable_cards, pot, roundIn, currplayer, game.totalWealth)
        lstmOut = []
        lstmIndex = 0
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
            lstmIndex = torch.argmax(softmax_output).item()
        #checking if it is the DQN turn
        move = None
        if currplayer['player'].get_name() == '5':
            stateTensor = pokerutils.convert_round_to_tesnor_DQN(game.players, viewable_cards, pot, roundIn, currplayer, lstmOut, game.totalWealth)
            stateTensor = stateTensor.to(device)
            print(stateTensor.shape)
            move = agent.play(stateTensor)
            state_action_pairs.append((stateTensor, move, currplayer["player"].get_money(), pot, roundIn ))
            if move == 0:
                move = "fold"
            elif move == 1:
                move = "c"
            elif move == 2:
                move = "raise"
        else:
            if lstmIndex == 0:
                move = "fold"
            elif lstmIndex == 1 or lstmIndex == 2:
                move = "c"
            elif lstmIndex == 3 or lstmIndex == 4:
                move = "raise"
        gameStatus, currplayer, players, pot, cardStatus, roundIn = game.next_turn(move)

    print("game finished " + str(epoch))
    for player in players:
        if player["player"].get_name() == "5":
            endReward = player['Q']
            endResult = player['result']
            for i in range(len(state_action_pairs)):
                state, action, bankroll, pot, roundIn = state_action_pairs[i]
                done = i == len(state_action_pairs) - 1
                next_state = torch.zeros_like(state) if done else state_action_pairs[i + 1][0]
                reward = 0
                if done:
                    reward = endReward
                else:
                    reward = pokerutils.calculate_reward(bankroll, action, endResult, pot, roundIn, total_wealth)
                reward *= (10 ** 6)
                # print(reward)
                agent.update_replay_buffer(state, action, next_state, done, reward)
    agent.train(batch_size)

agent.save('./models/DQN_V_LSTM.pth')