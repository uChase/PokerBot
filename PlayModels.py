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
batch_size = 16

playerList = []
models = []
agents = []
valid_options = ["1", "2", "3", "4", "5", "6"]
print(
    "There are 9 slots for players. You can choose to have a human player in one of the slots. If you do not want any more players, enter 5 in the slot. I reccomend you enter 1, 2, 2, 3, 5, 6, as a game with 5 players is easiest to follow on the terminal."
)
total_wealth = 0
for i in range(9):
    while True:
        userInput = input(
            f"Player {i+1}: Enter 1 for adapted DQN, 2 for LSTM (emulated player), 3 for raw DQN (not adapted), 4 for learning DQN (model that learns against others, experimental, very high learn rate), 5 for human player (you), 6 for no one in this slot:  "
        )
        if userInput in valid_options:
            break
        else:
            print("Invalid option. Please try again.")
    if userInput == "6":
        break
    if userInput == "1":
        models.append(DQN1.DQN(input_dim, output_dim).to(device))
        models[i].load_state_dict(
            torch.load("./models/DQN_V_LSTM.pth", map_location=device)
        )
        models[i].eval()
        agents.append("null")
    elif userInput == "2":
        models.append("LSTM")
        agents.append("null")
    elif userInput == "3":
        models.append(DQN1.DQN(input_dim, output_dim).to(device))
        models[i].load_state_dict(
            torch.load("./models/DQN_4_2.pth", map_location=device)
        )
        models[i].eval()
        agents.append("null")
    elif userInput == "4":
        models.append("agent")
        agents.append(
            DQN1.PokerAgent(input_dim, output_dim, 100000, 0.4, 0.99, 1.0, 0.01, 0.8)
        )
    elif userInput == "5":
        models.append("human")
        agents.append("null")
    else:
        models.append("none")
    playerList.append(pokergame.Player(str(i + 1), 1000))
    total_wealth += 1000

print(
    "Starting game...\n You have 3 there are three availible moves: fold, c, and raise. \n fold is folding obviously\n c is either check or call depending on context \n raise will raise 15$ \n you will be asked to input your move in the form of a number 1, 2, or 3 \n 1 for fold, 2 for c, 3 for raise"
)
finished = False

while not finished:
    players = []
    game = pokergame.PokerGame()
    for player in playerList:
        game.add_player(player)
    enough = game.check_enough()
    if not enough:
        print("need more players")
        break
    gameOver = False
    initialPlayerMoney = [player.get_money() for player in playerList]
    currplayer, cardStatus, community_cards, pot, roundIn = game.startGame()
    viewable_cards = []
    firstRound = True
    state_action_pairs = {agent_id: [] for agent_id in range(len(agents))}
    while not gameOver:
        (
            gameStatus,
            currplayer,
            players,
            pot,
            cardStatus,
            roundIn,
        ) = game.check_round_redo()
        if firstRound:
            pOrder = [player["player"].get_name() for player in players]
            print("player order:", pOrder)
            print("Button: Player ", pOrder[0])
            print("Small Blind($1): Player ", pOrder[1])
            print("Big Blind($2): Player ", pOrder[2])
            firstRound = False
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
                print("*** FLOP ***:", viewable_cards)
            elif cardStatus == 2:
                viewable_cards = community_cards[:4]
                print("*** TURN ***:", viewable_cards)
            elif cardStatus == 3:
                viewable_cards = community_cards[:5]
                print("*** RIVER ***:", viewable_cards)
        if models[int(currplayer["player"].get_name()) - 1] == "human":
            print("Your cards:", currplayer["holeCards"])
            print("Community cards:", viewable_cards)
            print("Your money: $", currplayer["player"].get_money())
            print("Pot: $", pot)
            print("Current bet: $", roundIn)
            print("Your amount to call: $", roundIn - currplayer["roundIn"])
            print("Your move options, 1 for fold, 2 for c, 3 for raise")

            while True:
                userInput = input("Enter your move: ")
                if userInput in valid_options:
                    break
                else:
                    print("Invalid option. Please try again.")
            if userInput == "1":
                (
                    gameStatus,
                    currplayer,
                    players,
                    pot,
                    cardStatus,
                    roundIn,
                ) = game.next_turn("fold")
            elif userInput == "2":
                (
                    gameStatus,
                    currplayer,
                    players,
                    pot,
                    cardStatus,
                    roundIn,
                ) = game.next_turn("c")
            elif userInput == "3":
                (
                    gameStatus,
                    currplayer,
                    players,
                    pot,
                    cardStatus,
                    roundIn,
                ) = game.next_turn("raise")
            else:
                print("Invalid option. Please try again.")
        else:
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
            if models[int(currplayer["player"].get_name()) - 1] == "LSTM":
                if lstmIndex == 0:
                    print(f"Player {currplayer['player'].get_name()} folded")
                    (
                        gameStatus,
                        currplayer,
                        players,
                        pot,
                        cardStatus,
                        roundIn,
                    ) = game.next_turn("fold")
                elif lstmIndex == 1 or lstmIndex == 2:
                    print(f"Player {currplayer['player'].get_name()} called")
                    (
                        gameStatus,
                        currplayer,
                        players,
                        pot,
                        cardStatus,
                        roundIn,
                    ) = game.next_turn("c")
                elif lstmIndex == 3 or lstmIndex == 4:
                    print(f"Player {currplayer['player'].get_name()} raised $15")
                    (
                        gameStatus,
                        currplayer,
                        players,
                        pot,
                        cardStatus,
                        roundIn,
                    ) = game.next_turn("raise")
            elif models[int(currplayer["player"].get_name()) - 1] == "agent":
                stateTensor = pokerutils.convert_round_to_tesnor_DQN(
                    game.players, viewable_cards, pot, roundIn, currplayer, lstmOut
                )
                stateTensor = stateTensor.to(device)
                agent_id = int(currplayer["player"].get_name()) - 1
                agent = agents[agent_id]
                move = agent.play(stateTensor)
                state_action_pairs[agent_id].append(
                    (stateTensor, move, currplayer["player"].get_money(), pot, roundIn)
                )
                max_index = move
                if max_index == 0:
                    move = "fold"
                    print(f"Player {currplayer['player'].get_name()} folded")
                elif max_index == 1:
                    move = "c"
                    print(f"Player {currplayer['player'].get_name()} called")
                elif max_index == 2:
                    move = "raise"
                    print(f"Player {currplayer['player'].get_name()} raised $15")
                (
                    gameStatus,
                    currplayer,
                    players,
                    pot,
                    cardStatus,
                    roundIn,
                ) = game.next_turn(move)
            else:
                stateTensor = pokerutils.convert_round_to_tesnor_DQN(
                    game.players, viewable_cards, pot, roundIn, currplayer, lstmOut
                )
                stateTensor = stateTensor.to(device)
                q_values = models[int(currplayer["player"].get_name()) - 1](
                    stateTensor
                ).to(device)
                action = torch.argmax(q_values).item()
                max_index = action
                move = ""
                if max_index == 0:
                    move = "fold"
                    print(f"Player {currplayer['player'].get_name()} folded")
                elif max_index == 1:
                    move = "c"
                    print(f"Player {currplayer['player'].get_name()} called")
                elif max_index == 2:
                    move = "raise"
                    print(f"Player {currplayer['player'].get_name()} raised $15")
                (
                    gameStatus,
                    currplayer,
                    players,
                    pot,
                    cardStatus,
                    roundIn,
                ) = game.next_turn(move)
    print("Game over")
    for player in playerList:
        for p in players:
            if p["player"].get_name() == player.get_name():
                print(f"Player {player.get_name()} had {p['holeCards']}")
        if player.get_money() >= initialPlayerMoney[int(player.get_name()) - 1]:
            print(
                f"Player {player.get_name()} won ${player.get_money() - initialPlayerMoney[int(player.get_name())-1]}"
            )
        else:
            print(
                f"Player {player.get_name()} lost ${initialPlayerMoney[int(player.get_name())-1] - player.get_money()}"
            )
    for player in players:
        agent_id = int(player["player"].get_name()) - 1
        agent = agents[agent_id]
        if agent == "null":
            continue
        endReward = player["Q"]
        endResult = player["result"]
        for i in range(len(state_action_pairs[agent_id])):
            state, action, bankroll, pot, roundIn = state_action_pairs[agent_id][i]
            done = i == len(state_action_pairs[agent_id]) - 1
            next_state = (
                torch.zeros_like(state)
                if done
                else state_action_pairs[agent_id][i + 1][0]
            )
            reward = 0
            if done:
                reward = endReward
            else:
                reward = pokerutils.calculate_reward(
                    bankroll, action, endResult, pot, roundIn, total_wealth
                )
            reward *= 10**6
            # print(reward)
            agent.update_replay_buffer(state, action, next_state, done, reward)
    for i, agent in enumerate(agents):
        if agent == "null":
            continue
        agent.train(batch_size)
        print("Trained Player ", i + 1)
    print("Would you like to play again? (y/n)")
    while True:
        userInput = input()
        if userInput == "y":
            break
        elif userInput == "n":
            finished = True
            break
        else:
            print("Invalid option. Please try again.")
